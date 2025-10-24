from __future__ import annotations
from typing import Any, List
from .base import Strategy
from .context import StrategyContext
from .requirement_adaptive import RequirementAdaptiveStrategy
from .predictive import PredictiveStrategy
from .incremental import IncrementalStrategy

class LiteratureMixStrategy(Strategy):
    """
    论文式混合器：按 SLA 压力把窗口预算切给 自适应(Req-Adaptive)/Predictive/Incremental。
    - P95 越接近阈值 → 自适应与 Predictive 占比越高；
    - 其余给 Incremental 兜底。
    """
    name = "literature_mix"

    def __init__(self):
        self.req_adp = RequirementAdaptiveStrategy()
        self.pred = PredictiveStrategy()
        self.incr = IncrementalStrategy()

    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        ids: List[Any] = []
        B = ctx.window_budget

        sla_pressure = 0.0
        if ctx.p95_ms is not None and ctx.p95_target_ms:
            sla_pressure = min(1.0, ctx.p95_ms / ctx.p95_target_ms)

        # 简单启发式配比：P95 压力越大，越偏向自适应/预测
        wb = int(B * (0.35 * sla_pressure))      # Requirement-Adaptive
        wp = int(B * (0.25 * sla_pressure + 0.1))# Predictive
        wi = B - wb - wp                         # Incremental

        if wb > 0:
            tmp = StrategyContext(**{**ctx.__dict__, "window_budget": wb})
            ids.extend(self.req_adp.select_ids(tmp))
        if wp > 0:
            tmp = StrategyContext(**{**ctx.__dict__, "window_budget": wp})
            ids.extend(self.pred.select_ids(tmp))
        if wi > 0:
            tmp = StrategyContext(**{**ctx.__dict__, "window_budget": wi})
            ids.extend(self.incr.select_ids(tmp))

        # 去重并裁剪
        seen, uniq = set(), []
        for _id in ids:
            if _id not in seen:
                seen.add(_id)
                uniq.append(_id)
            if len(uniq) >= B:
                break
        return uniq
