"""Students add their own algorithm modules here.

导入各算法模块即完成向 recommender.contracts.PROVIDERS 的注册。
实验一：content（基于内容的推荐）已接入。
实验二：popular（热门）、cold_start（冷启动）、itemcf（物品协同过滤）已接入；
als 等待后续实验实现。
"""
from . import cold_start, content, itemcf, popular  # noqa: F401  (注册各 Provider)
