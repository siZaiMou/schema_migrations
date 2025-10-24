from __future__ import annotations
from typing import Any, List
from .base import Strategy
from .context import StrategyContext

class EagerStrategy(Strategy):
    """Eager：全集回填，读路径无 on-read，Debt≈0，但发布成本高。"""
    name = "eager"

    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        q = {ctx.version_field: {"$ne": ctx.target_version}}
        return [d["_id"] for d in ctx.coll.find(q, {"_id": 1}).limit(ctx.window_budget)]
