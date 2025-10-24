from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pymongo.collection import Collection

@dataclass
class StrategyContext:
    coll: Collection
    version_field: str = "_sv"      # 文档的 schemaVersion 字段
    target_version: str = "k+1"     # 目标版本号

    # 每窗口预算（数量或抽象成本单位；由策略解释）
    window_budget: int = 50_000

    # 评分/画像集合（Predictive / 自适应使用）
    access_stats_coll: str = "access_stats"  # {_id, p, w, ts_last?, ...}

    # —— 观测指标（由 APM/埋点写入，策略可参考） ——
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    on_release_cost: Optional[float] = None
    on_read_cost: Optional[float] = None
    debt_ratio: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    # SLA / 约束
    p95_target_ms: Optional[float] = None
    cost_window_budget: Optional[float] = None

    # Predictive / 自适应参数
    ps_size: int = 10_000           # 预测集大小（本窗口上限）
    es_alpha: float = 0.5           # 指数平滑系数（若你在生成 access_stats 时用得到）
