"""离线评价数据加载：从冻结评价文件读取训练与标签（仅 evaluation 模式使用）。

- evaluation/train.csv：训练交互（构建各算法的离线模型/画像，也是 validation 的排除集）
- evaluation/validation.csv / test.csv / cold-kN-support.csv：相关标签（rating>=4.0 为正例）
不直接连业务库，保证离线评价只用冻结 train 拟合（课程硬性要求）。
"""
from functools import lru_cache

from recommender.data import FrozenData
from recommender.business import Feedback, UserHistory


@lru_cache(maxsize=None)
def _rows(path: str) -> list[tuple[str, str, float]]:
    """path 形如 'evaluation/train.csv'，返回 [(uid, movie_id, rating), ...]。"""
    data = FrozenData()
    out = []
    for row in data.rows(path, evaluation=True):
        out.append((str(row["userId"]), str(row["movieId"]), float(row["rating"])))
    return out


def interactions(path: str) -> dict[str, list[tuple[str, float]]]:
    """uid -> [(movie_id, rating), ...]。"""
    acc: dict[str, list[tuple[str, float]]] = {}
    for uid, mid, rating in _rows(path):
        acc.setdefault(uid, []).append((mid, rating))
    return acc


def labels(path: str) -> dict[str, set[str]]:
    """uid -> {movie_id}（rating>=4.0 的正例集合）。"""
    acc: dict[str, set[str]] = {}
    for uid, mid, rating in _rows(path):
        if rating >= 4.0:
            acc.setdefault(uid, set()).add(mid)
    return acc


def user_items_from(path: str) -> dict[str, list[str]]:
    """uid -> [movie_id]。"""
    acc: dict[str, list[str]] = {}
    for uid, mid, _ in _rows(path):
        acc.setdefault(uid, []).append(mid)
    return acc


@lru_cache(maxsize=None)
def train_interactions() -> dict[str, list[tuple[str, float]]]:
    return interactions("evaluation/train.csv")


@lru_cache(maxsize=None)
def train_stats() -> tuple[dict[str, tuple[int, float]], float]:
    stats: dict[str, tuple[int, float]] = {}
    total_c = 0
    total_s = 0.0
    for uid, mid, rating in _rows("evaluation/train.csv"):
        c, s = stats.get(mid, (0, 0.0))
        stats[mid] = (c + 1, s + rating)
        total_c += 1
        total_s += rating
    return stats, (total_s / total_c if total_c else 0.0)


@lru_cache(maxsize=None)
def train_cooccurrence() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, int]]:
    item_users: dict[str, list[str]] = {}
    user_items: dict[str, list[str]] = {}
    for uid, mid, _ in _rows("evaluation/train.csv"):
        item_users.setdefault(mid, []).append(uid)
        user_items.setdefault(uid, []).append(mid)
    item_n = {m: len(us) for m, us in item_users.items()}
    return item_users, user_items, item_n


def train_user_history(uid: str) -> UserHistory:
    """把某个用户的 train 交互包装成 UserHistory，供算法按离线训练数据推荐。"""
    rows = tuple(Feedback(m, r) for m, r in train_interactions().get(uid, []))
    return UserHistory("ml:" + uid, rows, ())


def validation_interactions() -> dict[str, list[tuple[str, float]]]:
    return interactions("evaluation/validation.csv")
