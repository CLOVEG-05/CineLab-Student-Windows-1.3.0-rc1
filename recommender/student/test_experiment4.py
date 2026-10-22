"""实验四 S1 测试：加权 RRF 融合推荐的契约、来源追溯与真实效果。

运行：python -m recommender.student.test_experiment4
环境：宿主机无业务库，统一切换到 evaluation 数据源与清单（与 evaluate.py 同机制）。
"""
import sys

import recommender.student as _student  # noqa: F401  触发 Provider 注册
import recommender.business as business
from recommender.contracts import Algorithm, PROVIDERS, RecommendationRequest, recommend
from recommender.data import DATASET, SPLIT, FrozenData
import recommender.evaluation as ev

from recommender.student import ratings, eval_data, popular, content, itemcf, als, rrf

EVAL_PATHS = {
    Algorithm.popular: popular.EVAL_MANIFEST_PATH,
    Algorithm.content: content.EVAL_MANIFEST_PATH,
    Algorithm.itemcf: itemcf.EVAL_MANIFEST_PATH,
    Algorithm.als: als.EVAL_MANIFEST_PATH,
    Algorithm.rrf: rrf.EVAL_MANIFEST_PATH,
}


def use_eval_env():
    """切换到 evaluation 数据源与清单（含 rrf 的全部组件）。"""
    switch = [(PROVIDERS[Algorithm.rrf], EVAL_PATHS[Algorithm.rrf])]
    switch += [(PROVIDERS[c], EVAL_PATHS[c]) for c in rrf.COMPONENTS]
    originals = [(p, p.model_manifest_path) for p, _ in switch]
    for p, path in switch:
        p.model_manifest_path = str(path)
    ratings.fetch_stats = eval_data.train_stats
    ratings.load_cooccurrence = eval_data.train_cooccurrence
    business.read_history = lambda req: eval_data.train_user_history(str(req.subject_id))
    popular.popular_ranking.cache_clear()
    return originals


def restore_env(originals):
    import importlib
    importlib.reload(business)  # 恢复原 read_history（覆盖测试期间的补丁）
    for p, orig in originals:
        p.model_manifest_path = orig
    popular.popular_ranking.cache_clear()


def req(uid: int, limit: int = 10) -> RecommendationRequest:
    return RecommendationRequest("ml", int(uid), limit, "evaluation", DATASET, SPLIT)


def check(name: str, cond: bool, detail: str = ""):
    print(("  PASS " if cond else "  FAIL ") + name + (("  | " + detail) if detail else ""))
    if not cond:
        raise AssertionError(name)


def main() -> int:
    print("=== 实验四 S1：加权 RRF 融合测试 ===")
    check("rrf 已注册到 PROVIDERS", Algorithm.rrf in PROVIDERS)

    originals = use_eval_env()
    try:
        data = FrozenData()
        users = sorted(ev.eligible_users(data, "validation"), key=int)
        labels = eval_data.labels("evaluation/validation.csv")

        # --- 契约与来源追溯 ---
        uid = users[0]
        result = recommend(Algorithm.rrf, req(uid))
        items = result.items
        check("返回条数 <= 10", len(items) <= 10, f"n={len(items)}")
        check("名次从 1 连续", [it.rank for it in items] == list(range(1, len(items) + 1)))
        check("无重复电影", len({it.movie_id for it in items}) == len(items))
        rated = {m for m, _ in eval_data.train_interactions().get(str(uid), [])}
        check("排除已评分电影", not ({it.movie_id for it in items} & rated))
        check("模型版本正确", result.model_version == rrf.MODEL_VERSION)
        check("算法字段为 rrf", all(it.algorithm == Algorithm.rrf for it in items))

        all_sources = set()
        for it in items:
            check(f"#{it.rank} 有来源列表", bool(it.recall_sources))
            check(f"#{it.rank} 来源名次为正整数",
                  bool(it.recall_ranks) and all(isinstance(v, int) and v >= 1
                                                for v in it.recall_ranks.values()))
            check(f"#{it.rank} 来源在融合流内",
                  set(it.recall_sources) <= {c.value for c in rrf.COMPONENTS},
                  ",".join(it.recall_sources))
            # 加权 RRF 分数必须等于 Σ w_s/(k+rank_s)，且被 Σw_s/(k+1) 限界（证明未混加原始分）。
            expect = round(sum(rrf.STREAM_WEIGHTS[Algorithm[n]] / (rrf.RRF_K + r)
                               for n, r in it.recall_ranks.items()), 6)
            check(f"#{it.rank} 分数符合加权 RRF 公式", abs((it.score or 0.0) - expect) < 1e-9,
                  f"score={it.score} expect={expect}")
            bound = sum(rrf.STREAM_WEIGHTS.values()) / (rrf.RRF_K + 1)
            check(f"#{it.rank} 分数在 RRF 理论界内", 0 < (it.score or 0) <= bound)
            all_sources.update(it.recall_sources)
        check("整体至少两路真实候选被融合", len(all_sources) >= 2, ",".join(sorted(all_sources)))

        # --- 个性化：两个不同用户结果不同 ---
        r2 = recommend(Algorithm.rrf, req(users[7]))
        top1 = {it.movie_id for it in items}
        top2 = {it.movie_id for it in r2.items}
        check("不同用户推荐结果不同", top1 != top2, f"jaccard={len(top1 & top2)}/{len(top1)}")

        # --- 确定性：重复调用一致 ---
        r_again = recommend(Algorithm.rrf, req(uid))
        check("重复调用结果一致", [it.movie_id for it in items] == [it.movie_id for it in r_again.items])

        # --- 样本对比：融合 vs 单路（200 用户） ---
        n = min(200, len(users))
        metrics = {name: [0.0, 0.0] for name in ("popular", "content", "itemcf", "als", "rrf")}
        for u in users[:n]:
            rel = labels.get(str(u), set())
            for name in metrics:
                res = recommend(Algorithm[name], req(u))
                hits = len({it.movie_id for it in res.items} & rel)
                metrics[name][0] += hits / 10
                metrics[name][1] += hits / len(rel) if rel else 0.0
        print(f"\n  样本 {n} 用户对比（融合 vs 单路）：")
        print(f"  {'算法':<10}{'P@10':>10}{'R@10':>10}")
        for name, (p, r) in metrics.items():
            print(f"  {name:<10}{p / n:>10.4f}{r / n:>10.4f}")
        check("rrf 在样本上有非空结果且指标为正", metrics["rrf"][0] / n > 0)
        print("\n=== 实验四 S1 测试全部通过 ===")
        return 0
    finally:
        restore_env(originals)


if __name__ == "__main__":
    sys.exit(main())
