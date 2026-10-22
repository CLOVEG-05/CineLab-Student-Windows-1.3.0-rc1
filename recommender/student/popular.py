"""实验二：Popular 热门推荐。

策略（利用历史评分数量 + 平均评分）：
- 从业务库 history_ratings 聚合每部电影的评分数 v 与平均分 R，及全量平均分 C。
- 采用 IMDb 加权评分避免小样本高分霸榜：
      WR = v/(v+m) * R + m/(v+m) * C     （m=50 先验平滑项）
- 排序：WR 降序；同分按 movie_id 升序；取最多 limit(<=10) 部，名次从 1 连续。
- 真实数据、完全确定性（无随机）；冷启动（cold_start）复用本排名作兜底。
"""
from datetime import datetime, timezone
from functools import lru_cache

from recommender.contracts import (
    Algorithm,
    PROVIDERS,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
)
from recommender.data import FrozenData
from recommender.student.manifest_util import make_manifests
from recommender.student import ratings
import recommender.business as business

MODEL_VERSION = "popular-bayesian-v1"
MIN_VOTES = 50

_ARTIFACT = {"strategy": "imdb-weighted-rating", "min_votes_prior": MIN_VOTES,
             "inputs": ["history_ratings(movie_id,rating)"]}
_PARAMS = {"ranking": "weighted-rating", "min_votes_prior": MIN_VOTES,
           "tie_break": "movie_id-asc", "max_results": 10}
MODEL_MANIFEST_PATH, EVAL_MANIFEST_PATH = make_manifests(
    "popular", MODEL_VERSION, "popular-params.json", _ARTIFACT, _PARAMS)


def _catalog() -> frozenset[str]:
    return frozenset(FrozenData().catalog())


@lru_cache(maxsize=1)
def popular_ranking() -> list[tuple[str, float]]:
    """全局热门排名：[(movie_id, WR), ...] 按 WR 降序、movie_id 升序。"""
    stats, mean_c = ratings.fetch_stats()
    catalog = _catalog()
    scored = []
    for movie_id, (votes, avg) in stats.items():
        if movie_id not in catalog or votes <= 0:
            continue
        weight = votes / (votes + MIN_VOTES)
        wr = weight * avg + (1.0 - weight) * mean_c
        scored.append((movie_id, round(wr, 6)))
    scored.sort(key=lambda kv: (-kv[1], int(kv[0])))
    return scored


def build_result(request: RecommendationRequest, ranking: list[tuple[str, float]],
                 exclude: frozenset[str], algorithm: Algorithm,
                 model_version: str) -> RecommendationResult:
    picked = [(m, s) for m, s in ranking if m not in exclude][: request.limit]
    items = [
        RecommendationItem(
            movie_id=movie_id,
            rank=rank,
            algorithm=algorithm,
            model_version=model_version,
            score=score,
        )
        for rank, (movie_id, score) in enumerate(picked, start=1)
    ]
    return RecommendationResult(
        algorithm=algorithm,
        model_version=model_version,
        dataset_version=request.dataset_version,
        split_version=request.split_version,
        runtime_mode=request.runtime_mode,
        generated_at=datetime.now(timezone.utc),
        items=items,
    )


class PopularRecommender:
    model_manifest_path = str(MODEL_MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        history = business.read_history(request)
        rated_ids = frozenset(fb.movie_id for fb in history.ratings)
        # 排除用户已评分电影（线上避免重复推荐；离线满足候选排除协议）。
        return build_result(request, popular_ranking(), rated_ids,
                            Algorithm.popular, MODEL_VERSION)


# 注册到现有 Provider 机制（导入本模块即完成注册）。
PROVIDERS[Algorithm.popular] = PopularRecommender()
