"""实验二 popular / itemcf / cold_start 单元测试（TDD，免数据库）。

通过 monkeypatch 替换 ratings.fetch_stats / ratings.load_cooccurrence 与
business.read_history，验证：
- 三个 Provider 均已注册且通过 recommender.contracts 全部契约校验；
- popular：加权评分排序、确定性、真实目录、名次连续；
- itemcf：排除已评分电影、不同用户结果个性化不同；
- cold_start：零评分用户得到热门兜底（合理推荐），已评分用户排除已看。

运行：
    python -m recommender.student.test_experiment2
    python -m pytest recommender/student/test_experiment2.py
"""
import recommender.business as business
from recommender.business import Feedback, UserHistory
from recommender.contracts import Algorithm, PROVIDERS, RecommendationRequest, recommend
from recommender.student import popular, ratings

DATASET = "rec-course-v3-train5-1.0.0"
SPLIT = "temporal-v1-cold500-train5-v3"


def _request(subject_id: int = 14, kind: str = "ml", limit: int = 10) -> RecommendationRequest:
    return RecommendationRequest(kind, subject_id, limit, "business", DATASET, SPLIT)


def _history(request, ratings_pairs):
    rows = tuple(Feedback(m, r) for m, r in ratings_pairs)
    return UserHistory(f"{request.subject_kind}:{request.subject_id}", rows, ())


# ---- 伪造的聚合统计（movie_id -> (评分数, 平均分)）与全量平均分 ----
FAKE_STATS = ({"1": (200, 4.2), "10": (300, 3.9), "500": (5, 5.0), "600": (100, 4.0)}, 3.8)

# ---- 伪造共现结构：电影 -> 评分用户集合 ----
FAKE_ITEM_USERS = {
    "1": ["ml:14", "ml:55", "ml:7"],
    "10": ["ml:14", "ml:7"],
    "20": ["ml:55"],
    "100": ["ml:7", "ml:55"],
    "200": ["ml:14", "ml:7"],
    "300": ["ml:55", "ml:7"],
}


def _fake_cooccurrence():
    item_users = {m: list(us) for m, us in FAKE_ITEM_USERS.items()}
    user_items: dict[str, list[str]] = {}
    for movie, users in item_users.items():
        for user in users:
            user_items.setdefault(user, []).append(movie)
    item_n = {m: len(us) for m, us in item_users.items()}
    return item_users, user_items, item_n


def _install_fakes():
    popular.popular_ranking.cache_clear()
    original = (ratings.fetch_stats, ratings.load_cooccurrence, business.read_history)
    ratings.fetch_stats = lambda: FAKE_STATS
    ratings.load_cooccurrence = _fake_cooccurrence
    return original


def _restore(original):
    ratings.fetch_stats, ratings.load_cooccurrence, business.read_history = original
    popular.popular_ranking.cache_clear()


def _assert_valid(result, algorithm, model_version):
    assert result.algorithm == algorithm
    assert result.model_version == model_version
    assert result.dataset_version == DATASET and result.split_version == SPLIT
    assert 1 <= len(result.items) <= 10
    seen, prev = set(), None
    for rank, item in enumerate(result.items, start=1):
        assert item.rank == rank and item.movie_id not in seen
        seen.add(item.movie_id)
        assert item.score is not None
        assert item.algorithm == algorithm and item.model_version == model_version
        if prev is not None and item.score != prev:
            assert item.score < prev, "分数必须降序"
        prev = item.score
    return seen


def test_providers_registered():
    assert Algorithm.popular in PROVIDERS
    assert Algorithm.itemcf in PROVIDERS
    assert Algorithm.cold_start in PROVIDERS


def test_popular_valid_and_sorted():
    original = _install_fakes()
    try:
        result = recommend(Algorithm.popular, _request())
    finally:
        _restore(original)
    seen = _assert_valid(result, Algorithm.popular, "popular-bayesian-v1")
    # 小样本(5票)的高分电影不应压过数据充足的候选项之外的合法性：全部来自目录
    assert seen <= {"1", "10", "500", "600"}
    # 确定性：两次调用结果一致
    original = _install_fakes()
    try:
        again = recommend(Algorithm.popular, _request())
    finally:
        _restore(original)
    assert [i.movie_id for i in again.items] == [i.movie_id for i in result.items]


def test_itemcf_excludes_rated_and_personalizes():
    original = _install_fakes()
    try:
        business.read_history = lambda req, database=None: _history(
            req, (("1", "5.0"), ("10", "4.0"), ("20", "2.0")))
        result_14 = recommend(Algorithm.itemcf, _request(14))
        business.read_history = lambda req, database=None: _history(
            req, (("1", "4.0"), ("20", "3.0"), ("100", "5.0"), ("300", "4.0")))
        result_55 = recommend(Algorithm.itemcf, _request(55))
    finally:
        _restore(original)

    seen_14 = _assert_valid(result_14, Algorithm.itemcf, "itemcf-cosine-v1")
    # 排除全部已评分电影
    assert seen_14.isdisjoint({"1", "10", "20"})
    assert seen_14, "ml:14 应得到非空推荐"
    assert result_14.items[0].recall_sources  # 可解释来源

    seen_55 = _assert_valid(result_55, Algorithm.itemcf, "itemcf-cosine-v1")
    assert seen_55.isdisjoint({"1", "20", "100", "300"})
    # 个性化：两个历史用户的结果不同
    assert [i.movie_id for i in result_14.items] != [i.movie_id for i in result_55.items]


def test_itemcf_empty_history_returns_empty():
    original = _install_fakes()
    try:
        business.read_history = lambda req, database=None: _history(req, ())
        result = recommend(Algorithm.itemcf, _request(999, kind="app"))
    finally:
        _restore(original)
    assert result.items == []  # 无信号时诚实返回空，不伪造


def test_cold_start_zero_rating_falls_back_to_popular():
    original = _install_fakes()
    try:
        business.read_history = lambda req, database=None: _history(req, ())
        cold = recommend(Algorithm.cold_start, _request(3, kind="app"))
        pop = recommend(Algorithm.popular, _request(3, kind="app"))
    finally:
        _restore(original)
    seen = _assert_valid(cold, Algorithm.cold_start, "cold-start-popular-v1")
    # 零评分用户的热门兜底与全局热门一致（合理推荐）
    assert [i.movie_id for i in cold.items] == [i.movie_id for i in pop.items]
    assert seen == {i.movie_id for i in pop.items}


def test_cold_start_excludes_rated():
    original = _install_fakes()
    try:
        business.read_history = lambda req, database=None: _history(req, (("1", "5.0"),))
        cold = recommend(Algorithm.cold_start, _request(14))
    finally:
        _restore(original)
    seen = _assert_valid(cold, Algorithm.cold_start, "cold-start-popular-v1")
    assert "1" not in seen, "已评分电影必须被排除"


def _run_script():
    tests = [test_providers_registered, test_popular_valid_and_sorted,
             test_itemcf_excludes_rated_and_personalizes,
             test_itemcf_empty_history_returns_empty,
             test_cold_start_zero_rating_falls_back_to_popular,
             test_cold_start_excludes_rated]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n全部 {len(tests)} 项测试通过。")


if __name__ == "__main__":
    _run_script()
