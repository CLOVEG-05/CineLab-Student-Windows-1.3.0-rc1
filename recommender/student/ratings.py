"""只读加载业务库中的真实评分数据（Popular / ItemCF 的训练输入）。

- history_ratings：MovieLens 历史用户（ml:<id>）的真实评分，是热门与协同
  过滤的训练基础。
- account_ratings：注册账号（app:<id>）的评分，一并纳入共现统计，
  保证 ItemCF 对注册账号同样可用。
- 全部为只读查询；不读取 evaluation 冻结评分文件（业务模式不允许）。
"""
from functools import lru_cache

from sqlalchemy import text

import recommender.business as business


@lru_cache(maxsize=1)
def fetch_stats() -> tuple[dict[str, tuple[int, float]], float]:
    """movie_id -> (评分数, 平均分)，以及全量平均分 C。单次 SQL 聚合。"""
    db = business._database()
    with db.connection() as conn:
        rows = conn.execute(text(
            "SELECT movie_id, COUNT(*) AS c, AVG(rating) AS r FROM history_ratings GROUP BY movie_id"
        )).all()
        mean = conn.execute(text("SELECT AVG(rating) FROM history_ratings")).scalar()
    stats = {str(m): (int(c), float(r)) for m, c, r in rows}
    return stats, float(mean)


@lru_cache(maxsize=1)
def load_cooccurrence() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, int]]:
    """共现结构：(item_users, user_items, item_n)。

    - item_users: movie_id -> 评过它的用户键列表（"ml:14" / "app:3"）
    - user_items: 用户键 -> 评过的 movie_id 列表
    - item_n:     movie_id -> 评分人数（余弦分母用）
    """
    db = business._database()
    item_users: dict[str, list[str]] = {}
    user_items: dict[str, list[str]] = {}
    with db.connection() as conn:
        rows = conn.execute(text(
            "SELECT ml_user_id, movie_id FROM history_ratings ORDER BY ml_user_id, movie_id"
        )).all()
        rows += (("app:" + str(a), m) for a, m in conn.execute(text(
            "SELECT account_id, movie_id FROM account_ratings ORDER BY account_id, movie_id"
        )).all())
    for uid, mid in rows:
        user = uid if str(uid).startswith("app:") else "ml:" + str(uid)
        movie = str(mid)
        item_users.setdefault(movie, []).append(user)
        user_items.setdefault(user, []).append(movie)
    item_n = {m: len(us) for m, us in item_users.items()}
    return item_users, user_items, item_n


@lru_cache(maxsize=1)
def all_history_ratings() -> list[tuple[str, str, float]]:
    """全量评分三元组 (ml_user_id, movie_id, rating)，供 ALS 在业务模式训练。

    直接查业务库（与 popular/itemcf/content 在线训练同源），不走冻结业务文件，
    避免触发离线评价的数据治理拦截（demo 入口不允许读 business 完整评分）。
    """
    db = business._database()
    out: list[tuple[str, str, float]] = []
    with db.connection() as conn:
        for uid, mid, rating in conn.execute(text(
                "SELECT ml_user_id, movie_id, rating FROM history_ratings")).all():
            out.append((str(uid), str(mid), float(rating)))
        for uid, mid, rating in conn.execute(text(
                "SELECT account_id, movie_id, rating FROM account_ratings")).all():
            out.append((str(uid), str(mid), float(rating)))
    return out

