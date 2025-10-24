from __future__ import annotations
from typing import Any, List
from pymongo import DESCENDING
from .base import Strategy
from .context import StrategyContext
from .incremental import IncrementalStrategy

class PredictiveStrategy(Strategy):
    """
    Predictive：使用热度/最近访问分数构成“预测集”，每窗迁前 ps_size 个。
    access_stats 结构建议：{ _id, p, w, ts_last? }；缺失则自动回退到增量。
    """
    name = "predictive"

    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        stats = ctx.coll.database.get_collection(ctx.access_stats_coll)
        cur = stats.find({}, {"_id": 1, "p": 1, "w": 1}) \
                   .sort([("w", DESCENDING), ("p", DESCENDING)]) \
                   .limit(ctx.ps_size * 2)  # 取多一点再过滤
        cand = [s["_id"] for s in cur]
        if not cand:
            return IncrementalStrategy().select_ids(ctx)

        q = {"_id": {"$in": cand}, ctx.version_field: {"$ne": ctx.target_version}}
        cap = min(ctx.ps_size, ctx.window_budget)
        return [d["_id"] for d in ctx.coll.find(q, {"_id": 1}).limit(cap)]
