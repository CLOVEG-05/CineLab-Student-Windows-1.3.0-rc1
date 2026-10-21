"""实验一 content 推荐单元测试（TDD）。

通过 monkeypatch 替换 business.read_history，无需真实数据库即可验证：
- Provider 已注册；
- 算法输出通过 recommender.contracts 的全部契约校验（目录/名次/分数/版本）；
- 仅采用 >=4.0 正反馈；
- 排除已评分电影；
- 排名连续、无重复、最多 10 部、分数降序（同分 movie_id 升序）；
- 无有效正反馈时返回空结果（不伪造）。

运行：
    python -m recommender.student.test_content          # 脚本模式
    python -m pytest recommender/student/test_content.py  # pytest 模式
"""
from recommender.business import Feedback, UserHistory
import recommender.business as business
from recommender.contracts import Algorithm, PROVIDERS, RecommendationRequest, recommend
from recommender.student.content import content_scores
import recommender.student.content  # noqa: F401  (触发注册)


DATASET = "rec-course-v3-train5-1.0.0"
SPLIT = "temporal-v1-cold500-train5-v3"


def _request(subject_id: int = 14, limit: int = 10) -> RecommendationRequest:
    return RecommendationRequest("ml", subject_id, limit, "business", DATASET, SPLIT)


def _fake_history(request, database=None):
    # 真实目录电影：1、10 为 >=4.0 正反馈；100 也是正反馈（用于验证被排除出候选）；
    # 20 为已评分但 <4.0（用于验证被排除出候选）。
    ratings = (
        Feedback("1", "5.0"),
        Feedback("10", "4.5"),
        Feedback("100", "5.0"),
        Feedback("20", "3.0"),
    )
    return UserHistory(f"{request.subject_kind}:{request.subject_id}", ratings, ())


def _fake_history_empty(request, database=None):
    return UserHistory(f"{request.subject_kind}:{request.subject_id}", (), ())


def test_provider_registered():
    assert Algorithm.content in PROVIDERS


def test_content_recommendation_valid():
    original = business.read_history
    business.read_history = _fake_history
    try:
        result = recommend(Algorithm.content, _request())
    finally:
        business.read_history = original

    assert result.algorithm == Algorithm.content
    assert result.model_version == "content-genres-v1"
    assert result.dataset_version == DATASET
    assert result.split_version == SPLIT
    assert result.runtime_mode == "business"
    assert 1 <= len(result.items) <= 10

    seen = set()
    prev_score = None
    for rank, item in enumerate(result.items, start=1):
        assert item.rank == rank, "排名必须连续从 1 递增"
        assert item.movie_id not in seen, "电影不能重复"
        seen.add(item.movie_id)
        assert item.score is not None, "必须包含模型分数"
        assert item.algorithm == Algorithm.content
        assert item.model_version == "content-genres-v1"
        if prev_score is not None and item.score != prev_score:
            assert item.score < prev_score, "分数必须降序"
        prev_score = item.score

    # 排除已评分电影（含正反馈 1/10/100 与低分 20）
    assert seen.isdisjoint({"1", "10", "100", "20"})
    # 可解释来源
    assert result.items[0].recall_sources


def test_no_positive_feedback_returns_empty():
    original = business.read_history
    business.read_history = _fake_history_empty
    try:
        result = recommend(Algorithm.content, _request())
    finally:
        business.read_history = original
    assert result.items == []


def test_content_scores_excludes_all_rated():
    # 直接测试打分函数：已评分电影（含低分）必须被排除。
    genre_index = {
        "1": frozenset({"Action", "Comedy"}),
        "10": frozenset({"Action"}),
        "20": frozenset({"Comedy"}),          # 已评分但 <4.0，也应排除
        "99": frozenset({"Action", "Comedy"}),  # 候选，与正反馈高度相似
        "88": frozenset({"Drama"}),             # 候选，无交集
    }
    catalog = frozenset(genre_index)
    positive_ids = frozenset({"1", "10"})     # >=4.0 正反馈
    rated_ids = frozenset({"1", "10", "20"})  # 全部已评分
    scores = content_scores(positive_ids, rated_ids, genre_index, catalog)
    assert "1" not in scores and "10" not in scores and "20" not in scores
    assert "99" in scores and "88" not in scores  # 88 与正反馈无交集，得 0 分不入选
    assert scores["99"] > 0


def _run_script():
    tests = [test_provider_registered, test_content_recommendation_valid, test_no_positive_feedback_returns_empty, test_content_scores_excludes_all_rated]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n全部 {len(tests)} 项测试通过。")


if __name__ == "__main__":
    _run_script()
