"""实验二：ItemCF 基于物品的协同过滤。

原理：
- 把每部电影表示为"评过它的用户集合"（来自业务库全量真实评分，ml+app）。
- 电影相似度 = 用户集合的余弦相似度：
      sim(i, c) = |U_i ∩ U_c| / sqrt(|U_i| * |U_c|)
- 用户对候选 c 的打分 = 其每条历史评分按分值加权后对 c 的相似度累加：
      score(c) = Σ_{i ∈ 用户已评分} (rating_i / 5) * sim(i, c)
  评分越高权重越大，因此低分电影对候选的拉动更弱（利用评分历史的方式）。
- 排除用户已评分的全部电影；分数降序、同分 movie_id 升序；最多 limit 部。
- 工程实现：按 (i → U_i → 各用户的其他电影) 遍历累积 co_acc[c] += w_i/√n_i，
  最终 score(c) = co_acc[c]/√n_c，避免显式两两比较 8459 部电影。
"""
import math
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
from recommender.student.manifest_util import build_manifest
from recommender.student import ratings
import recommender.business as business

MODEL_VERSION = "itemcf-cosine-v1"

MODEL_MANIFEST_PATH = build_manifest(
    "itemcf",
    MODEL_VERSION,
    "itemcf-params.json",
    {"similarity": "cosine-of-user-sets", "weighting": "rating/5",
     "inputs": ["history_ratings", "account_ratings"]},
    {"similarity": "cosine", "weighting": "rating/5", "exclude_rated": True,
     "tie_break": "movie_id-asc", "max_results": 10},
)


def _catalog() -> frozenset[str]:
    return frozenset(FrozenData().catalog())


def itemcf_scores(weights: dict[str, float], rated_ids: frozenset[str],
                  item_users: dict[str, list[str]], user_items: dict[str, list[str]],
                  item_n: dict[str, int], catalog: frozenset[str]) -> dict[str, float]:
    """为每部未评分候选累加评分加权的物品相似度（见模块 docstring 公式）。"""
    acc: dict[str, float] = {}
    for item_id, weight in weights.items():
        n_i = item_n.get(item_id)
        if not n_i:
            continue
        base = weight / math.sqrt(n_i)
        for user in item_users.get(item_id, ()):
            for other in user_items.get(user, ()):
                if other in rated_ids or other == item_id:
                    continue
                acc[other] = acc.get(other, 0.0) + base
    return {movie: value / math.sqrt(item_n[movie])
            for movie, value in acc.items()
            if movie in catalog and item_n.get(movie)}


def build_recommendation(request: RecommendationRequest, weights: dict[str, float],
                         rated_ids: frozenset[str], co: tuple) -> RecommendationResult:
    item_users, user_items, item_n = co
    scores = {movie: round(value, 6) for movie, value in
              itemcf_scores(weights, rated_ids, item_users, user_items, item_n, _catalog()).items()}
    # 先舍入再排序：保证"同分按 movie_id 升序"的契约校验稳定成立。
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], int(kv[0])))
    items = [
        RecommendationItem(
            movie_id=movie_id,
            rank=rank,
            algorithm=Algorithm.itemcf,
            model_version=MODEL_VERSION,
            score=round(score, 6),
            recall_sources=sorted(rated_ids),
        )
        for rank, (movie_id, score) in enumerate(ranked[: request.limit], start=1)
    ]
    return RecommendationResult(
        algorithm=Algorithm.itemcf,
        model_version=MODEL_VERSION,
        dataset_version=request.dataset_version,
        split_version=request.split_version,
        runtime_mode=request.runtime_mode,
        generated_at=datetime.now(timezone.utc),
        items=items,
    )


class ItemCfRecommender:
    model_manifest_path = str(MODEL_MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        history = business.read_history(request)
        rated_ids = frozenset(fb.movie_id for fb in history.ratings)
        # 权重 = 评分/5：评分越高，与其相似的电影获得的加分越大。
        weights = {fb.movie_id: float(fb.rating) / 5.0 for fb in history.ratings}
        return build_recommendation(request, weights, rated_ids, ratings.load_cooccurrence())


# 注册到现有 Provider 机制（导入本模块即完成注册）。
PROVIDERS[Algorithm.itemcf] = ItemCfRecommender()
