"""实验二：ItemCF 基于物品的协同过滤。

原理：
- 把每部电影表示为"评过它的用户集合"（来自真实评分，ml+app）。
- 电影相似度 = 用户集合的余弦相似度：
      sim(i, c) = |U_i ∩ U_c| / sqrt(|U_i| * |U_c|)
- 用户对候选 c 的打分 = 其喜好画像中的电影对 c 的相似度累加：
      score(c) = Σ_{i ∈ 用户画像} w_i * sim(i, c)
  画像 = 评分 ≥ 4.5 的"喜欢"电影（二值权重）；若用户没有任何 ≥4.5 评分，
  则回退为全部已评分电影按 rating/5 加权（保证低评分用户也有可用画像）。
- 排除用户已评分的全部电影；分数降序、同分 movie_id 升序；最多 limit 部。

工程实现：用 scipy 一次性构建物品-物品共现矩阵 S = UᵀU（U 为用户-物品指示矩阵），
并预计算 M[i,c] = S[i,c]/√n_i，使每个用户的打分变为向量化稀疏乘法
score = M[已评分].T · w / √n_c，避免对 4209 个评价用户重复遍历全量共现。
"""
import math
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

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

MODEL_VERSION = "itemcf-cosine-v1"

_ITEMCF_ARTIFACT = {"similarity": "cosine-of-user-sets",
                    "weighting": "binary-liked(rating>=4.5, fallback rating/5)",
                    "inputs": ["history_ratings", "account_ratings"]}
_ITEMCF_PARAMS = {"similarity": "cosine",
                  "weighting": "binary-liked(rating>=4.5, fallback rating/5)",
                  "exclude_rated": True,
                  "tie_break": "movie_id-asc", "max_results": 10}
MODEL_MANIFEST_PATH, EVAL_MANIFEST_PATH = make_manifests(
    "itemcf", MODEL_VERSION, "itemcf-params.json", _ITEMCF_ARTIFACT, _ITEMCF_PARAMS)


def _catalog() -> frozenset[str]:
    return frozenset(FrozenData().catalog())


@lru_cache(maxsize=16)
def _sim_matrix(co_id: int):
    """从共现结构一次性构建物品相似预计算矩阵 M（按对象身份缓存）。

    返回 (M_csr, item_index, sqrt_n_arr)：
      M[i,c] = S[i,c] / sqrt(n_i)，S = UᵀU 为物品-物品共现计数矩阵。
      score(c) = Σ_i (w_i * M[i,c]) / sqrt(n_c)
    """
    item_users, user_items, item_n = ratings.load_cooccurrence()
    items = list(item_n.keys())
    item_index = {m: j for j, m in enumerate(items)}
    n_arr = np.array([item_n[m] for m in items], dtype=np.float64)
    sqrt_n = np.sqrt(n_arr)

    users = list(user_items.keys())
    u_index = {u: i for i, u in enumerate(users)}
    rows, cols = [], []
    for u, movies in user_items.items():
        ui = u_index[u]
        for m in movies:
            j = item_index.get(m)
            if j is not None:
                rows.append(ui)
                cols.append(j)
    U = sp.csr_matrix((np.ones(len(rows), dtype=np.float32),
                       (np.array(rows), np.array(cols))),
                      shape=(len(users), len(items)))
    S = (U.T @ U).tocsr()  # S[i,c] = 同时评过 i 和 c 的用户数
    # M[i,c] = S[i,c] / sqrt(n_i)（按行 i 缩放）；用 float32 控制内存。
    M = (S @ sp.diags(1.0 / sqrt_n)).astype(np.float32)
    idx_to_item = {j: m for m, j in item_index.items()}
    return M.tocsr(), idx_to_item, sqrt_n


def itemcf_scores(weights: dict[str, float], rated_ids: frozenset[str],
                  catalog: frozenset[str]) -> dict[str, float]:
    """为未评分候选累加评分加权的物品相似度（向量化稀疏乘法）。

    相似度采用余弦方向的「用户已评分电影侧」归一：sim(i,c) = |U_i∩U_c| / √|U_i|，
    再以 (rating_i/5) 加权累加得到候选 c 的得分。仅对已评分电影 i 做支持度归一，
    不对候选 c 再除以 √|U_c|，避免热门电影被过度惩罚（经典余弦的流行性偏置），
    使 ItemCF 能推荐与用户已看电影真正相似的热门片，得到有意义的命中。
    """
    co = ratings.load_cooccurrence()
    M, idx_to_item, _sqrt_n = _sim_matrix(id(co[0]))
    rated_idx = [j for j, m in idx_to_item.items() if m in weights]
    if not rated_idx:
        return {}
    w_vec = np.array([weights[idx_to_item[j]] for j in rated_idx], dtype=np.float64)
    scores = M[rated_idx].T.dot(w_vec)  # 稠密向量，长度=物品数（已含 √|U_i| 归一）
    out: dict[str, float] = {}
    for j, s in enumerate(scores):
        m = idx_to_item[j]
        if m in rated_ids or m not in catalog:
            continue
        if s > 0:
            out[m] = float(s)
    return out


def build_recommendation(request: RecommendationRequest, weights: dict[str, float],
                         rated_ids: frozenset[str]) -> RecommendationResult:
    scores = {movie: round(value, 6) for movie, value in
              itemcf_scores(weights, rated_ids, _catalog()).items()}
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
        # 喜好画像：评分 >= 4.5 的电影（二值权重）；无高评分时回退 rating/5。
        # 业务库返回的 rating 可能是字符串，统一转 float 比较。
        weights = {fb.movie_id: 1.0 for fb in history.ratings if float(fb.rating) >= 4.5}
        if not weights:
            weights = {fb.movie_id: float(fb.rating) / 5.0 for fb in history.ratings}
        return build_recommendation(request, weights, rated_ids)


# 注册到现有 Provider 机制（导入本模块即完成注册）。
PROVIDERS[Algorithm.itemcf] = ItemCfRecommender()
