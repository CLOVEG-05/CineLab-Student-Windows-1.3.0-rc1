"""实验四 S1：加权 RRF（Reciprocal Rank Fusion）多算法融合推荐。

目标（课程手册实验四 S1）：
- 融合至少两路真实候选；去重、过滤、排序；显示推荐来源及名次。
- **不能直接混加不同量纲的原始分数**：各算法原始分（余弦相似度、加权评分、
  ALS 重构分、隐因子点积）量纲完全不同，因此采用 RRF——只用"名次"融合：
      RRF(d) = Σ_{s ∈ 融合流} 1 / (k + rank_s(d))     （k=60，标准取值）
- 每个推荐项带 `recall_sources`（贡献该结果的算法列表）与 `recall_ranks`
  （各算法中的名次），实现来源可追溯。

融合流：als / itemcf / content / popular（四路真实候选；popular 常非空，
保证零评分用户也至少有两路真实候选，避免空结果或伪造个性化）。
"""
from datetime import datetime, timezone

from recommender.contracts import (
    Algorithm,
    PROVIDERS,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
    recommend,
)
from recommender.data import FrozenData
from recommender.student.manifest_util import make_manifests
import recommender.business as business

MODEL_VERSION = "rrf-rank-fusion-v1"

# RRF 平滑常数 k（标准取值 60）；每路候选深度。
RRF_K = 60
STREAM_TOPN = 20

# 各路名次权重（加权 RRF）：按实验三全量 validation 的 Precision@10 比例设定
# （als 0.0615 / itemcf 0.0520 / popular 0.0338 / content 0.0040，以最强路归一），
# 表现越好的算法在名次融合中话语权越大；权重只作用于名次，不混加原始分。
STREAM_WEIGHTS = {Algorithm.als: 1.0, Algorithm.itemcf: 0.85,
                  Algorithm.popular: 0.55, Algorithm.content: 0.07}

# 融合的真实算法流（顺序仅用于文档展示，融合与顺序无关）。
COMPONENTS: tuple[Algorithm, ...] = tuple(STREAM_WEIGHTS)

_ARTIFACT = {
    "fusion": "weighted-reciprocal-rank-fusion",
    "k": RRF_K,
    "stream_topn": STREAM_TOPN,
    "components": [c.value for c in COMPONENTS],
    "stream_weights": {c.value: STREAM_WEIGHTS[c] for c in COMPONENTS},
    "weight_basis": "实验三全量 validation Precision@10 比例归一（als=1.0）",
    "inputs": ["als", "itemcf", "content", "popular 真实候选"],
    "note": "仅加权融合名次，不混加不同量纲原始分；来源与名次写入 recall_sources/recall_ranks",
}
_PARAMS = {
    "fusion": "weighted-reciprocal-rank-fusion", "k": RRF_K, "stream_topn": STREAM_TOPN,
    "components": [c.value for c in COMPONENTS],
    "stream_weights": {c.value: STREAM_WEIGHTS[c] for c in COMPONENTS},
    "dedupe": True, "exclude_rated": True, "tie_break": "movie_id-asc",
    "max_results": 10, "min_streams": 2,
}
MODEL_MANIFEST_PATH, EVAL_MANIFEST_PATH = make_manifests(
    "rrf", MODEL_VERSION, "rrf-params.json", _ARTIFACT, _PARAMS)


def _catalog() -> frozenset[str]:
    return frozenset(FrozenData().catalog())


def fuse(request: RecommendationRequest, exclude: frozenset[str],
         limit: int) -> list[dict]:
    """融合多路真实候选，返回按 RRF 分降序的候选（含来源与名次）。"""
    # 每路取前 STREAM_TOPN 个真实候选（各自已完成排除已评分/目录过滤）。
    ranks: dict[str, dict[str, int]] = {}  # movie_id -> {算法: 名次}
    for comp in COMPONENTS:
        sub = RecommendationRequest(
            subject_kind=request.subject_kind,
            subject_id=request.subject_id,
            limit=STREAM_TOPN,
            runtime_mode=request.runtime_mode,
            dataset_version=request.dataset_version,
            split_version=request.split_version,
        )
        result = recommend(comp, sub)  # 走完整契约（真实候选）
        for item in result.items:
            ranks.setdefault(item.movie_id, {})[comp.value] = item.rank

    # 去重已在 ranks 聚合时完成；过滤已评分电影与目录外电影。
    catalog = _catalog()
    scored = []
    for movie_id, comp_ranks in ranks.items():
        if movie_id in exclude or movie_id not in catalog:
            continue
        # 加权 RRF：score = Σ_s w_s / (k + rank_s)，权重只作用于名次。
        score = round(sum(STREAM_WEIGHTS[Algorithm[name]] / (RRF_K + rank)
                          for name, rank in comp_ranks.items()), 6)
        # 来源按名次升序、再按算法名排序，保证确定性展示。
        ordered = sorted(comp_ranks.items(), key=lambda kv: (kv[1], kv[0]))
        scored.append({
            "movie_id": movie_id,
            "score": score,
            "recall_sources": [name for name, _ in ordered],
            "recall_ranks": dict(ordered),
        })

    # RRF 分降序；同分按 movie_id 升序（满足 check_result 同分排序契约）。
    scored.sort(key=lambda d: (-d["score"], int(d["movie_id"])))
    return scored[:limit]


class RrfRecommender:
    model_manifest_path = str(MODEL_MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        history = business.read_history(request)
        rated_ids = frozenset(fb.movie_id for fb in history.ratings)
        picked = fuse(request, rated_ids, request.limit)
        items = [
            RecommendationItem(
                movie_id=d["movie_id"],
                rank=rank,
                algorithm=Algorithm.rrf,
                model_version=MODEL_VERSION,
                score=d["score"],
                recall_sources=d["recall_sources"],
                recall_ranks=d["recall_ranks"],
            )
            for rank, d in enumerate(picked, start=1)
        ]
        return RecommendationResult(
            algorithm=Algorithm.rrf,
            model_version=MODEL_VERSION,
            dataset_version=request.dataset_version,
            split_version=request.split_version,
            runtime_mode=request.runtime_mode,
            generated_at=datetime.now(timezone.utc),
            items=items,
        )


# 注册到现有 Provider 机制（导入本模块即完成注册）。
PROVIDERS[Algorithm.rrf] = RrfRecommender()
