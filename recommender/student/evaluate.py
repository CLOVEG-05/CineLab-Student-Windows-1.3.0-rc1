"""实验三离线评价脚本：统一协议下计算各算法的 Precision@10 / Recall@10。

协议（严格遵循 recommender/evaluation.py 与实验手册）：
- 仅用冻结 evaluation/train.csv 拟合（绝不用业务库全量历史）。
- 候选 = catalog 减去当前分区要求的已训练/已评分电影。
- 正例 = 该分区相关标签文件中 rating>=4.0 的电影。
- 每个算法对固定评价用户集合逐一生成 Top-10（空结果用户也须提交 items=[]，不得删除）。
- 产出 predictions(JSONL) + EvaluationResult(JSON)，并用课程 check_predictions/check_evaluation_result 校验。

用法：
    python -m recommender.student.evaluate --partition validation --max-users 50
    python -m recommender.student.evaluate --partition validation
"""
import argparse
import json
import sys
from pathlib import Path

import recommender.student as _student  # noqa: F401  触发四个 Provider 注册
import recommender.business as business
from recommender.contracts import Algorithm, PROVIDERS, RecommendationRequest, recommend
from recommender.data import DATASET, SPLIT, FrozenData, sha256
import recommender.evaluation as ev

from recommender.student import ratings, eval_data
from recommender.student import popular, content, itemcf, als  # 实验三接入 als

# 各算法的 evaluation 清单路径（供 check_model 切换为评价模式）。
EVAL_PATHS = {
    Algorithm.popular: popular.EVAL_MANIFEST_PATH,
    Algorithm.content: content.EVAL_MANIFEST_PATH,
    Algorithm.itemcf: itemcf.EVAL_MANIFEST_PATH,
    Algorithm.als: als.EVAL_MANIFEST_PATH,
}


def _history_for(uid: str, partition: str) -> list[tuple[str, float]]:
    if partition == "validation":
        return eval_data.train_interactions().get(uid, [])
    if partition == "test":
        return eval_data.train_interactions().get(uid, []) + eval_data.validation_interactions().get(uid, [])
    return eval_data.interactions(f"evaluation/{partition}-support.csv").get(uid, [])


def _relevant_file(partition: str) -> str:
    if partition == "validation":
        return "evaluation/validation.csv"
    if partition == "test":
        return "evaluation/test.csv"
    return f"evaluation/{partition}-support.csv"


def run_algorithm(algorithm_name: str, partition: str, data: FrozenData,
                  limit: int = 10, max_users: int | None = None) -> dict:
    alg = Algorithm[algorithm_name]
    task = ev.task(data, partition)
    users = sorted(ev.eligible_users(data, partition), key=int)
    if max_users is not None:
        users = users[:max_users]
    provider = PROVIDERS[alg]
    orig_model_path = provider.model_manifest_path
    provider.model_manifest_path = str(EVAL_PATHS[alg])

    # 离线拟合：popular/itemcf 改用 train.csv 统计与共现；content 用 train 历史。
    prev_stats = ratings.fetch_stats
    prev_cooc = ratings.load_cooccurrence
    prev_history = business.read_history
    ratings.fetch_stats = eval_data.train_stats
    ratings.load_cooccurrence = eval_data.train_cooccurrence
    popular.popular_ranking.cache_clear()
    business.read_history = lambda req: eval_data.train_user_history(str(req.subject_id))

    predictions: list[dict] = []
    precisions: list[float] = []
    recalls: list[float] = []
    covered = 0
    rel_map = eval_data.labels(_relevant_file(partition))

    try:
        for uid in users:
            rel = rel_map.get(uid, set())
            request = RecommendationRequest("ml", int(uid), limit, "evaluation", DATASET, SPLIT)
            result = recommend(alg, request)  # 走完整契约校验（含 check_model 评价模式）
            if result.items:
                covered += 1
            top = [it.movie_id for it in result.items]
            hits = len(set(top) & rel)
            p = hits / limit if limit else 0.0
            r = hits / len(rel) if rel else 0.0
            precisions.append(p)
            recalls.append(r)
            predictions.append({"ml_user_id": str(uid), "result": result.model_dump(mode="json")})
    finally:
        ratings.fetch_stats = prev_stats
        ratings.load_cooccurrence = prev_cooc
        business.read_history = prev_history
        popular.popular_ranking.cache_clear()
        provider.model_manifest_path = orig_model_path

    precision_mean = sum(precisions) / len(precisions) if precisions else 0.0
    recall_mean = sum(recalls) / len(recalls) if recalls else 0.0
    coverage = covered / len(users) if users else 0.0

    eval_result = {
        "status": "completed",
        "runtime_mode": "evaluation",
        "algorithm": algorithm_name,
        "model_version": result.model_version,
        "task_sha256": task["task_sha256"],
        "model_manifest_sha256": sha256(Path(EVAL_PATHS[alg])),
        "dataset_version": DATASET,
        "split_version": SPLIT,
        "schema_version": task["schema_version"],
        "partition": partition,
        "user_count": len(users),
        "precision_at_10": round(precision_mean, 6),
        "recall_at_10": round(recall_mean, 6),
    }
    return {"eval_result": eval_result, "predictions": predictions, "precision": precision_mean,
            "recall": recall_mean, "coverage": coverage, "users": len(users)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="validation")
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--out", default="recommender/student/eval_results")
    args = ap.parse_args()

    data = FrozenData()
    task = ev.task(data, args.partition)
    print(f"分区 {args.partition}: 评价用户数={task['user_count']}, 任务 sha={task['task_sha256'][:12]}…")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ["popular", "content", "itemcf", "als"]:
        print(f"\n=== {name} ===")
        res = run_algorithm(name, args.partition, data, max_users=args.max_users)
        summary[name] = res["eval_result"]
        (out_dir / f"{name}-{args.partition}.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in res["predictions"]), encoding="utf-8")
        er_path = out_dir / f"{name}-{args.partition}-result.json"
        er_path.write_text(json.dumps(res["eval_result"], ensure_ascii=False, indent=2), encoding="utf-8")
        if args.max_users is None:  # 全量才做框架级校验（小样本用户数不符会失败）
            ev.check_predictions(str(out_dir / f"{name}-{args.partition}.jsonl"),
                                 Path(EVAL_PATHS[Algorithm[name]]), args.partition, data)
            ev.check_evaluation_result(str(er_path), Path(EVAL_PATHS[Algorithm[name]]), data)
        print(f"  Precision@10={res['precision']:.4f}  Recall@10={res['recall']:.4f}  "
              f"Coverage={res['coverage']:.4f}  用户={res['users']}")
    print("\n--- 对比表 ---")
    print(f"{'算法':<10}{'P@10':>10}{'R@10':>10}{'Coverage':>12}")
    for name, res in summary.items():
        er = res["eval_result"]
        print(f"{name:<10}{er['precision_at_10']:>10.4f}{er['recall_at_10']:>10.4f}{res['coverage']:>12.4f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
