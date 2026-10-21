"""实验共用：运行时生成合规 ModelManifest 与模型工件。

原因：课程部署脚本 deployment/build.ps1 构建 api 镜像时只复制
recommender/*.py，不打包 .json 工件；因此在模块导入时用冻结数据包装
(FrozenData) 的真实 sha 运行时生成清单与工件，保证容器内 check_model
始终可通过，且不依赖外部预生成文件。
"""
import json
import tempfile
from pathlib import Path

from recommender.data import DATASET, MANIFEST_SHA, SCHEMA, SPLIT, FrozenData, sha256

TOOL_VERSIONS = {"python": "3.11", "recommender": SCHEMA}


def build_manifest(algorithm: str, model_version: str, artifact_name: str,
                   artifact: dict, parameters: dict) -> Path:
    """生成 <artifact> 与 model_manifest.json，返回清单路径（check_model 读取）。"""
    base = Path(tempfile.gettempdir()) / ("cine_" + algorithm + "_model")
    base.mkdir(parents=True, exist_ok=True)
    (base / artifact_name).write_bytes(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest = sha256(base / artifact_name)
    files = FrozenData().manifest["files"]
    manifest = {
        "algorithm": algorithm,
        "model_version": model_version,
        "runtime_mode": "business",
        "dataset_version": DATASET,
        "split_version": SPLIT,
        "schema_version": SCHEMA,
        "manifest_sha256": MANIFEST_SHA,
        "training_files": {
            "business/history_ratings.csv": files["business/history_ratings.csv"]["sha256"]
        },
        "feature_files": {
            "movies.csv": files["movies.csv"]["sha256"],
            "training-catalog.csv": files["training-catalog.csv"]["sha256"],
        },
        "artifacts": {artifact_name: digest},
        "parameters": parameters,
        "tool_versions": TOOL_VERSIONS,
    }
    path = base / "model_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
