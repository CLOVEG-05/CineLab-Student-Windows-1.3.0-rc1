"""实验一：基于内容的电影推荐（content）。

设计要点（对齐《推荐算法及应用》实验一）：
- 内容特征：MovieLens 电影的 `genres` 字段，转为 multi-hot / 类型集合特征。
- 用户画像：仅采用评分 >= 4.0 的历史正反馈电影（候选阶段排除用户全部已评分电影）。
- 相似度：电影类型集合的余弦相似度 = 共同类型数 / sqrt(|A| * |B|)。
- 候选打分：对目录中每部未评分候选，累加它与用户每部正反馈电影的余弦相似度。
- 排序：分数降序；同分时按 movie_id 升序；取最多 limit(<=10) 部，排名从 1 连续递增。
- 可解释：recall_sources 记录参与打分的正反馈电影。
- 所有输出均通过 recommender.contracts 的 check_result 契约校验。

模型清单与工件（content-features.json）在模块导入时**运行时生成**到可写目录，
原因：课程部署脚本 `build.ps1` 构建 api 镜像时只复制 `recommender/*.py`，
不会打包 `.json` 工件；运行时生成可保证容器内的 `check_model` 始终能通过，
且不依赖外部预生成的文件。
"""
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import recommender.business as business
from recommender.contracts import (
    Algorithm,
    PROVIDERS,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
)
from recommender.data import (
    DATASET,
    MANIFEST_SHA,
    SCHEMA,
    SPLIT,
    FrozenData,
)

from recommender.student.manifest_util import make_manifests

MODEL_VERSION = "content-genres-v1"
POSITIVE_THRESHOLD = 4.0

_ARTIFACT_NAME = "content-features.json"


def _genre_vocabulary() -> list[str]:
    """从冻结 movies.csv 汇总全部出现过的电影类型（确定性，用于工件）。"""
    data = FrozenData()
    vocab: set[str] = set()
    for row in data.rows("movies.csv"):
        for g in (row.get("genres") or "").split("|"):
            if g:
                vocab.add(g)
    return sorted(vocab)


def _content_artifact() -> dict:
    return {"feature": "genres-multi-hot", "genre_vocabulary": _genre_vocabulary()}


_CONTENT_PARAMS = {
    "feature": "genres-multi-hot",
    "similarity": "cosine",
    "positive_rating_threshold": 4.0,
    "aggregation": "sum-over-positive-movies",
    "exclude_rated": True,
    "max_results": 10,
}
MODEL_MANIFEST_PATH, EVAL_MANIFEST_PATH = make_manifests(
    "content", MODEL_VERSION, _ARTIFACT_NAME, _content_artifact(), _CONTENT_PARAMS)


@lru_cache(maxsize=1)
def _catalog() -> frozenset[str]:
    return frozenset(FrozenData().catalog())


@lru_cache(maxsize=1)
def _genre_index() -> dict[str, frozenset[str]]:
    """movie_id -> 类型集合（来自冻结 movies.csv，仅读取特征文件，不接触评价数据）。"""
    data = FrozenData()
    index: dict[str, frozenset[str]] = {}
    for row in data.rows("movies.csv"):
        tokens = frozenset(g for g in (row.get("genres") or "").split("|") if g)
        index[row["movie_id"]] = tokens
    return index


def _cosine(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) * len(b)) ** 0.5


def content_scores(positive_ids: frozenset[str], rated_ids: frozenset[str],
                   genre_index: dict[str, frozenset[str]], catalog: frozenset[str]) -> dict[str, float]:
    """为每部未评分候选累加其与正反馈电影的余弦相似度。

    候选阶段排除用户已经评分过的全部电影（不只正反馈），满足实验“排除已评分电影”。
    """
    scores: dict[str, float] = {}
    for cand in catalog:
        if cand in rated_ids:
            continue
        g_cand = genre_index.get(cand)
        if not g_cand:
            continue
        total = 0.0
        for pid in positive_ids:
            g_pos = genre_index.get(pid)
            if g_pos:
                total += _cosine(g_cand, g_pos)
        if total > 0.0:
            scores[cand] = total
    return scores


def build_recommendation(request: RecommendationRequest, positive_ids: frozenset[str],
                         rated_ids: frozenset[str], genre_index: dict[str, frozenset[str]],
                         catalog: frozenset[str]) -> RecommendationResult:
    scores = {movie: round(value, 6) for movie, value in content_scores(
        positive_ids, rated_ids, genre_index, catalog).items()}
    # 先舍入再排序：保证"同分按 movie_id 升序"的契约校验稳定成立。
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], int(kv[0])))
    items = [
        RecommendationItem(
            movie_id=movie_id,
            rank=rank,
            algorithm=Algorithm.content,
            model_version=MODEL_VERSION,
            score=round(score, 6),
            recall_sources=list(positive_ids),
        )
        for rank, (movie_id, score) in enumerate(ranked[: request.limit], start=1)
    ]
    return RecommendationResult(
        algorithm=Algorithm.content,
        model_version=MODEL_VERSION,
        dataset_version=request.dataset_version,
        split_version=request.split_version,
        runtime_mode=request.runtime_mode,
        generated_at=datetime.now(timezone.utc),
        items=items,
    )


class ContentRecommender:
    model_manifest_path = str(MODEL_MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        catalog = _catalog()
        genre_index = _genre_index()
        history = business.read_history(request)
        rated_ids = frozenset(fb.movie_id for fb in history.ratings)
        positive_ids = frozenset(
            fb.movie_id for fb in history.ratings if float(fb.rating) >= POSITIVE_THRESHOLD
        )
        return build_recommendation(request, positive_ids, rated_ids, genre_index, catalog)


# 注册到现有 Provider 机制（导入本模块即完成注册）。
PROVIDERS[Algorithm.content] = ContentRecommender()
