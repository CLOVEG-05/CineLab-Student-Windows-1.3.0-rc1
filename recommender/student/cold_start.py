"""实验二：Cold Start 用户冷启动。

策略（对齐实验要求"Cold Start 在零评分条件下能够提供合理推荐"）：
- 新注册观众（零评分、零收藏）没有任何个人信号，此时采用全局热门排名
  （popular 的 IMDb 加权评分）作为冷启动兜底，给出合理的 Top-10。
- 若用户已有评分（少量评分用户），同样给出热门兜底结果，但排除其已评分
  电影，避免重复推荐看过的内容。
- 全部来自真实评分聚合，确定性输出，不使用固定/随机列表。

（选做的"选择兴趣类型"未实现，属加分项，不影响验收。）
"""
from recommender.contracts import (
    Algorithm,
    PROVIDERS,
    RecommendationRequest,
    RecommendationResult,
)
from recommender.student.manifest_util import build_manifest
from recommender.student.popular import build_result, popular_ranking
import recommender.business as business

MODEL_VERSION = "cold-start-popular-v1"

MODEL_MANIFEST_PATH = build_manifest(
    "cold_start",
    MODEL_VERSION,
    "cold-start-strategy.json",
    {"strategy": "popular-fallback",
     "description": "zero-signal users fall back to global popular ranking",
     "inputs": ["history_ratings(movie_id,rating)"]},
    {"strategy": "popular-fallback", "exclude_rated": True,
     "tie_break": "movie_id-asc", "max_results": 10},
)


class ColdStartRecommender:
    model_manifest_path = str(MODEL_MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        history = business.read_history(request)
        rated_ids = frozenset(fb.movie_id for fb in history.ratings)
        # 零评分用户 rated_ids 为空集 → 纯全局热门兜底；已评分类用户则排除已看。
        return build_result(request, popular_ranking(), rated_ids,
                            Algorithm.cold_start, MODEL_VERSION)


# 注册到现有 Provider 机制（导入本模块即完成注册）。
PROVIDERS[Algorithm.cold_start] = ColdStartRecommender()
