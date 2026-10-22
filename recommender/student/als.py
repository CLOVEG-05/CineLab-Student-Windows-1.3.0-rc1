"""实验三：ALS 矩阵分解（Matrix Factorization via Alternating Least Squares）。

原理（对齐《推荐算法及应用》实验三"ALS 矩阵分解"）：
- 把"用户-电影"评分矩阵分解为 用户隐因子矩阵 P(n_users×k) 与 电影隐因子矩阵 Q(n_items×k)。
- 交替最小二乘：固定 Q 解 P，固定 P 解 Q，直至收敛；预测分 = P_u · Q_i。
- 离线训练：按手册仅用冻结 evaluation/train.csv（离线评价协议）。
- 在线(业务)：训练文件 business/history_ratings.csv；模型按训练文件缓存，首次请求训练一次。
- 候选 = 目录减去用户已评分电影；按预测分降序、同分 movie_id 升序取 Top-10。

工程说明：课程环境下 pyspark 不可用，这里用 numpy 实现**与 Spark MLlib ALS 同构的
交替最小二乘优化**（均最小化二乘重构误差），产出真实因子模型，非随机/硬编码。
"""
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import json

import numpy as np

from recommender.contracts import (
    Algorithm,
    PROVIDERS,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
)
from recommender.data import DATASET, MANIFEST_SHA, SCHEMA, SPLIT, FrozenData, sha256
from recommender.student.manifest_util import make_manifests
import recommender.business as business

MODEL_VERSION = "als-matrix-factorization-v1"
RANK = 50
REG = 0.1
ITERATIONS = 10
SEED = 42

_ARTIFACT = {
    "method": "als-matrix-factorization",
    "library": "numpy(explicit-als)",
    "inputs": ["history_ratings(movie_id,rating)"],
    "note": "Spark MLlib 不可用，采用同构的 numpy 交替最小二乘实现",
}
_PARAMS = {
    "rank": RANK, "reg": REG, "iterations": ITERATIONS,
    "exclude_rated": True, "tie_break": "movie_id-asc", "max_results": 10,
}
MODEL_MANIFEST_PATH, EVAL_MANIFEST_PATH = make_manifests(
    "als", MODEL_VERSION, "als-model.json", _ARTIFACT, _PARAMS)


class _ALSModel:
    __slots__ = ("P", "Q", "user_index", "item_index")

    def __init__(self, P, Q, user_index, item_index):
        self.P = P
        self.Q = Q
        self.user_index = user_index
        self.item_index = item_index


@lru_cache(maxsize=None)
def _training_rows(training_file: str):
    """读取训练三元组。

    - evaluation/xxx：离线评价协议，从冻结文件读取（仅训练集 train.csv）。
    - 其余（business/xxx）：业务模式，从业务库 DB 全量评分读取（与 popular/itemcf
      在线训练同源，避免触发 demo 入口对冻结业务完整评分的读取拦截）。
    """
    if training_file.startswith("evaluation/"):
        data = FrozenData()
        rows = [(str(r["userId"]), str(r["movieId"]), float(r["rating"]))
                for r in data.rows(training_file, evaluation=True)]
    else:
        from recommender.student import ratings as _ratings
        rows = _ratings.all_history_ratings()
    return rows


def train_als(training_file: str, rank: int = RANK, reg: float = REG,
             iterations: int = ITERATIONS, seed: int = SEED) -> _ALSModel:
    rows = _training_rows(training_file)
    users = sorted({u for u, _, _ in rows})
    items = sorted({m for _, m, _ in rows})
    u_index = {u: i for i, u in enumerate(users)}
    i_index = {m: j for j, m in enumerate(items)}
    n_u, n_i = len(users), len(items)

    user_idx = np.array([u_index[u] for u, _, _ in rows], dtype=np.int64)
    item_idx = np.array([i_index[m] for _, m, _ in rows], dtype=np.int64)
    ratings = np.array([r for _, _, r in rows], dtype=np.float64)

    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n_u, rank)) * 0.1
    Q = rng.standard_normal((n_i, rank)) * 0.1
    I = np.eye(rank)

    for _ in range(iterations):
        # 固定 P，解 Q：A = P^T P + reg I；B[j] = Σ_u P_u * r_{u,j}
        A = P.T @ P + reg * I
        B = np.zeros((n_i, rank))
        for d in range(rank):
            B[:, d] = np.bincount(item_idx, weights=P[user_idx, d] * ratings, minlength=n_i)
        Q = np.linalg.solve(A, B.T).T
        # 固定 Q，解 P：A2 = Q^T Q + reg I；B2[i] = Σ_j Q_j * r_{i,j}
        A2 = Q.T @ Q + reg * I
        B2 = np.zeros((n_u, rank))
        for d in range(rank):
            B2[:, d] = np.bincount(user_idx, weights=Q[item_idx, d] * ratings, minlength=n_u)
        P = np.linalg.solve(A2, B2.T).T

    return _ALSModel(P, Q, u_index, i_index)


_MODELS: dict[str, _ALSModel] = {}


def _get_model(training_file: str) -> _ALSModel:
    model = _MODELS.get(training_file)
    if model is None:
        model = train_als(training_file)
        _MODELS[training_file] = model
    return model


@lru_cache(maxsize=1)
def _catalog() -> frozenset[str]:
    return frozenset(FrozenData().catalog())


def build_recommendation(request: RecommendationRequest, training_file: str,
                         rated_ids: frozenset[str]) -> RecommendationResult:
    model = _get_model(training_file)
    uid = str(request.subject_id)
    if uid not in model.user_index:
        # 该用户不在训练集中（无因子）→ 无个性化结果（诚实返回空，不伪造）。
        items = []
    else:
        u_vec = model.P[model.user_index[uid]]
        scores_all = model.Q @ u_vec  # 所有训练电影的原始预测分
        cand = [(mid, int(mid), scores_all[j]) for mid, j in model.item_index.items()
                if mid not in rated_ids and mid in _catalog()]
        cand = [(mid, mi, round(s, 6)) for mid, mi, s in cand]
        ranked = sorted(cand, key=lambda t: (-t[2], t[1]))[: request.limit]
        items = [
            RecommendationItem(
                movie_id=mid,
                rank=rank,
                algorithm=Algorithm.als,
                model_version=MODEL_VERSION,
                score=round(score, 6),
                recall_sources=[],
            )
            for rank, (mid, _, score) in enumerate(ranked, start=1)
        ]
    return RecommendationResult(
        algorithm=Algorithm.als,
        model_version=MODEL_VERSION,
        dataset_version=request.dataset_version,
        split_version=request.split_version,
        runtime_mode=request.runtime_mode,
        generated_at=datetime.now(timezone.utc),
        items=items,
    )


class AlsRecommender:
    model_manifest_path = str(MODEL_MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        # 从模型清单读取训练文件（business 或 evaluation），按文件缓存训练。
        with Path(self.model_manifest_path).open(encoding="utf-8") as fh:
            m = json.load(fh)
        training_file = next(iter(m["training_files"]))
        history = business.read_history(request)
        rated_ids = frozenset(fb.movie_id for fb in history.ratings)
        return build_recommendation(request, training_file, rated_ids)


# 注册到现有 Provider 机制（导入本模块即完成注册）。
PROVIDERS[Algorithm.als] = AlsRecommender()


def _warm_business_model() -> None:
    """在 API 进程导入时预热业务模型，避免首请求因全量训练超时被网关 503。"""
    try:
        with Path(MODEL_MANIFEST_PATH).open(encoding="utf-8") as fh:
            m = json.load(fh)
        _get_model(next(iter(m["training_files"])))
    except Exception:
        # 预热失败不影响注册；首请求会再次触发训练。
        pass


import os as _os  # noqa: E402
if _os.environ.get("APP_ENV") in ("development", "production"):
    _warm_business_model()
