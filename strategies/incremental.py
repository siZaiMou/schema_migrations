from __future__ import annotations
from typing import Any, List, Optional, Dict
from .base import Strategy
from .context import StrategyContext

class IncrementalStrategy(Strategy):
    """Incremental：稳定分批清债，介于 eager / lazy 之间的折中。"""
    name = "incremental"

    def __init__(self, batch_hint: int = 5_000, base_query: Optional[Dict]=None):
        self.batch_hint = batch_hint
        self.base_query = base_query or {}

    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        q = {ctx.version_field: {"$ne": ctx.target_version}, **self.base_query}
        cap = min(ctx.window_budget, self.batch_hint)
        return [d["_id"] for d in ctx.coll.find(q, {"_id": 1}).limit(cap)]
