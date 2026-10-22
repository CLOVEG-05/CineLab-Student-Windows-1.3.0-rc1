"""Students add their own algorithm modules here.

导入各算法模块即完成向 recommender.contracts.PROVIDERS 的注册。
实验一：content（基于内容的推荐）已接入。
实验二：popular（热门）、cold_start（冷启动）、itemcf（物品协同过滤）已接入。
实验三：als（矩阵分解 ALS）已接入。
实验四：rrf（加权 RRF 多算法融合，S1）已接入。
"""
from . import als, cold_start, content, itemcf, popular, rrf  # noqa: F401  (注册各 Provider)
