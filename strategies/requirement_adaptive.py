from __future__ import annotations
from typing import Any, List, Optional
from .base import Strategy
from .context import StrategyContext
from .predictive import PredictiveStrategy

class RequirementAdaptiveStrategy(Strategy):
    """
    Requirement-Adaptive：基于“成本/延迟相对阈值”的均衡，动态调节 ps_size 后走 Predictive。
    算法：qc=c/cmax, ql=l/lmax, delta=(ql-qc)；ps_size *= (1 + rate*delta)。
    - ql 高 → 放大 ps（更积极提前回填，降低 on-read）；
    - qc 高 → 缩小 ps（控制发布成本）。
    """
    name = "requirement_adaptive"

    def __init__(self, adjust_rate: float = 0.3, min_ps: int = 1_000, max_ps: int = 300_000):
        self.adjust_rate = adjust_rate
        self.min_ps = min_ps
        self.max_ps = max_ps

    def adjust_prediction_set(self, ctx: StrategyContext) -> Optional[int]:
        c = 0.0
        if ctx.on_release_cost is not None:
            c += ctx.on_release_cost
        if ctx.on_read_cost is not None:
            c += ctx.on_read_cost
        cmax = ctx.cost_window_budget
        l = ctx.p95_ms
        lmax = ctx.p95_target_ms
        if not cmax or not lmax or l is None:
            return None

        qc = max(0.0, c / cmax)
        ql = max(0.0, l / lmax)
        delta = (ql - qc)
        scale = 1.0 + self.adjust_rate * delta
        new_ps = int(max(self.min_ps, min(self.max_ps, ctx.ps_size * scale)))
        if new_ps != ctx.ps_size:
            return new_ps
        return None

    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        new_ps = self.adjust_prediction_set(ctx)
        old = ctx.ps_size
        if new_ps:
            ctx.ps_size = new_ps
        try:
            return PredictiveStrategy().select_ids(ctx)
        finally:
            ctx.ps_size = old
