from __future__ import annotations
from typing import Any, List, Tuple
from .base import Strategy
from .context import StrategyContext

class BcoStrategy(Strategy):
    """
    收益/成本优化（Benefit-Cost Optimized）
    access_stats 要求（最小）：{ _id, p }
    若可用：{ _id, p, dL_ms, dC, c_io, c_cpu, c_idx }
    - 收益：b = α*(p*H*dL) + β*(p*H*dC)
    - 成本：c = w_io*c_io + w_cpu*c_cpu + w_idx*c_idx
    - 选择：按 b/c 排序，直到预算耗尽或数量上限
    """
    name = "bco"

    def __init__(self,
                 alpha_beta: Tuple[float,float]=(1.0,1.0),
                 cost_weights: Tuple[float,float,float]=(1.0,1.0,2.0),
                 horizon_hours: int = 24,
                 sample_multiplier: int = 3,
                 hard_cap: int = 200_000):
        self.alpha_beta = alpha_beta
        self.cost_weights = cost_weights
        self.horizon_hours = horizon_hours
        self.sample_multiplier = sample_multiplier
        self.hard_cap = hard_cap

    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        stats = ctx.coll.database.get_collection(ctx.access_stats_coll)
        α, β = self.alpha_beta
        w_io, w_cpu, w_idx = self.cost_weights
        H = max(1, self.horizon_hours)

        # 采样若干评分候选以减少排序成本
        sample_n = min(ctx.window_budget * self.sample_multiplier, self.hard_cap)
        cur = stats.find({}, {"_id":1, "p":1, "dL_ms":1, "dC":1, "c_io":1, "c_cpu":1, "c_idx":1}).limit(sample_n)

        scored = []
        for s in cur:
            p   = float(s.get("p", 0.0))
            dL  = float(s.get("dL_ms", 0.0))
            dC  = float(s.get("dC", 0.0))
            cio = float(s.get("c_io", 0.0))
            ccp = float(s.get("c_cpu", 0.0))
            cix = float(s.get("c_idx", 0.0))
            # 最小可用：没有成本画像时，把成本视作 1（退化为按收益排序）
            c = w_io*cio + w_cpu*ccp + w_idx*cix
            if c <= 0: c = 1.0
            b = α*(p*H*dL) + β*(p*H*dC)
            # 若无 dL/dC，也能跑（此时 b≈0 → 将靠 p 的微弱差异；建议至少提供 p）
            score = b / c if c > 0 else 0.0
            scored.append((s["_id"], score, c))

        if not scored:
            # 回退：没有画像则退化为增量
            from .incremental import IncrementalStrategy
            return IncrementalStrategy().select_ids(ctx)

        # 排序 & 按预算挑选
        scored.sort(key=lambda x: x[1], reverse=True)
        picked, used = [], 0.0
        # 这里把 window_budget 作为“成本预算”；如需数量预算，可改为 len(picked) < window_budget
        for _id, _score, cost in scored:
            if used + cost > ctx.window_budget:
                if picked: break
            picked.append(_id)
            used += cost
            if len(picked) >= ctx.window_budget:
                break

        # 过滤掉已迁完
        q = {"_id": {"$in": picked}, ctx.version_field: {"$ne": ctx.target_version}}
        return [d["_id"] for d in ctx.coll.find(q, {"_id":1}).limit(ctx.window_budget)]
