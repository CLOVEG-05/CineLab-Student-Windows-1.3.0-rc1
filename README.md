# 片刻 CineLab — 电影推荐系统（推荐算法课程大作业）

基于 **MovieLens 真实电影数据** 的电影推荐系统，课程框架技术栈：
**Python 3.12 · FastAPI · Vue 3 · Docker Compose · MySQL 8 · Pydantic · SQLAlchemy**。

学生在课程框架（CineLab Student 包）内，按实验手册逐个实现推荐算法 Provider，
注册进统一契约（`recommender/contracts.py` 的 `PROVIDERS`），并接入前端"推荐实验室"。

> ⚠️ 仓库不含课程冻结数据包（`data/packages/`）与密钥（`deployment/private/`），
> 二者已在 `.gitignore` 中排除，需按课程发放包本地放置。

---

## 当前完成进度

| 实验 | 内容 | 状态 |
|---|---|---|
| 实验一 | `content` 基于内容推荐（genres multi-hot + 余弦相似度） | ✅ 已接入 |
| 实验二 | `popular` 热门推荐 / `cold_start` 冷启动 / `itemcf` 物品协同过滤 | ✅ 已接入 |
| 实验三 | `als` 矩阵分解（numpy ALS）+ 四算法离线评价（P@10/R@10） | ✅ 已接入 |
| 实验四 | `rrf` 加权 RRF 多算法融合（S1） | ✅ 已接入 |
| 实验五 | `S3` Kafka 事件驱动实时推荐 | ⬜ 待实现 |

算法注册状态可随时通过 `GET /api/v1/experiments/algorithms` 查看。

---

## 目录结构

```
recommender/               # 推荐算法包（课程契约 + 学生实现）
  contracts.py             #   Algorithm 枚举 / DTO / PROVIDERS 注册表 / recommend() 契约校验
  validation.py            #   check_request / check_provider / check_model / check_result
  data.py                  #   冻结数据包装（manifest sha 校验、catalog 8459 部）
  business.py              #   只读用户历史适配器（ml → history_ratings，app → account_ratings）
  student/                 #   ★ 学生实现区
    content.py             #     实验一：基于内容推荐（content-genres-v1）
    popular.py             #     实验二：热门推荐（popular-bayesian-v1）
    cold_start.py          #     实验二：冷启动（cold-start-popular-v1，热门兜底）
    itemcf.py              #     实验二：物品协同过滤（itemcf-cosine-v1）
    als.py                 #     实验三：矩阵分解 ALS（numpy 交替最小二乘，als-matrix-factorization-v1）
    rrf.py                 #     实验四：加权 RRF 多算法融合（rrf-rank-fusion-v1）
    ratings.py             #     共用：业务库评分只读加载（聚合统计 / 共现结构 / ALS 全量三元组）
    manifest_util.py       #     共用：运行时生成合规 ModelManifest 与工件
    eval_data.py           #     实验三：评价数据加载（train/validation 冻结文件）
    evaluate.py            #     实验三：离线评价脚本（P@10/R@10，--algorithms 可选）
    test_content.py        #     实验一单元测试（4 项）
    test_experiment2.py    #     实验二单元测试（6 项）
    test_experiment4.py    #     实验四 S1 测试（RRF 契约/公式/来源追溯/个性化）
backend/                   # FastAPI 服务（课程提供，未改动业务接口格式）
frontend/                  # Vue 3 + Vite（红黑侧栏主题 + 推荐实验室页面）
deployment/                # Dockerfile / build.ps1 / compose 运行脚本（课程提供）
data/packages/…            # 课程冻结数据包（不入库）
```

---

## 算法原理速览

### 实验一 `content`（content-genres-v1）
- 电影特征：`genres` 按 `|` 切分为类型集合（multi-hot 等价）。
- 用户画像：历史评分 `≥4.0` 的正反馈电影。
- 相似度：`sim(A,B)=|A∩B|/√(|A||B|)`；候选得分 = 与各正反馈电影相似度之和。
- 排除用户全部已评分电影；分数降序、同分 `movie_id` 升序；Top-10。

### 实验二 `popular`（popular-bayesian-v1）
- 从业务库 `history_ratings` 聚合每部电影的评分数 `v` 与平均分 `R`、全量平均分 `C`。
- IMDb 加权评分抑制小样本霸榜：`WR = v/(v+m)·R + m/(v+m)·C`（`m=50`）。
- 纯全局、确定性；同分按 `movie_id` 升序。

### 实验二 `cold_start`（cold-start-popular-v1）
- 零评分/零收藏的新注册观众没有任何个人信号 → 采用全局热门排名兜底。
- 已有少量评分的用户 → 仍为热门兜底，但**排除其已评分电影**。
- （选做的"兴趣类型选择"未实现，属加分项。）

### 实验二 `itemcf`（itemcf-cosine-v1）
- 把每部电影表示为"评过它的用户集合"（业务库全量真实评分，ml + app）。
- 物品相似度 = 用户集合余弦：`sim(i,c)=|U_i∩U_c|/√(|U_i||U_c|)`。
- 用户对候选 `c` 的得分 = `Σ_{i∈已评分} (rating_i/5)·sim(i,c)`——评分越高拉动越强。
- 排除已评分电影；不同历史用户（如 ml:14 与 ml:55）得到完全不同的个性化列表。

### 契约与模型清单
每个 Provider 必须携带 `model_manifest.json`（`check_model` 校验硬编码版本号
`manifest_sha256/dataset/split/schema` + 训练/特征文件 sha + 真实存在的工件），
输出经 `check_result` 校验（合法目录、名次连续、无重复、同分 movie_id 升序）。
清单与工件由 `manifest_util.py` 在运行时生成到可写目录——因为课程构建脚本
`build.ps1` 只打包 `*.py`，这样容器内校验始终可通过。

---

## 本地运行与验证

```powershell
# 0) 准备：按课程发放包放置 data/packages/rec-course-v3-train5-1.0.0/ 与 deployment/private/

# 1) 构建学生代码镜像并启动（api + web）
powershell -ExecutionPolicy Bypass -File deployment\build.ps1 -Target all -Action Build
# 浏览器访问 http://127.0.0.1:8080

# 2) 单元测试（免数据库）
python -m recommender.student.test_content        # 实验一 4 项
python -m recommender.student.test_experiment2    # 实验二 6 项

# 3) 接口验证（真实数据）
#    算法注册状态
curl http://127.0.0.1:8080/api/v1/experiments/algorithms
#    历史用户（ml:14 / ml:55）各算法 Top-10
curl "http://127.0.0.1:8080/api/v1/demo/history-users/14/recommendations/content?limit=10"
curl "http://127.0.0.1:8080/api/v1/demo/history-users/14/recommendations/popular?limit=10"
curl "http://127.0.0.1:8080/api/v1/demo/history-users/14/recommendations/itemcf?limit=10"
curl "http://127.0.0.1:8080/api/v1/demo/history-users/55/recommendations/itemcf?limit=10"
#    冷启动：注册新账号并登录后
curl "http://127.0.0.1:8080/api/v1/me/recommendations/cold_start?limit=10"
```

页面验证：**推荐实验室** → 用户来源"MovieLens 历史用户" → 选 `ml:14`/`ml:55` →
切换 `content / popular / itemcf` 对比结果；"当前注册账号"零评分 → `cold_start`。

---

## 实验记录

详细实施记录（算法设计、TDD 过程、部署命令、真实接口验证证据、接手须知）见
`../实验记录/` 目录（实验一、实验二各一份）。

## 小组

CLOVEG-05 · 推荐算法课程大作业
