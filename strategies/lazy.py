from __future__ import annotations
from typing import Any, List
from .base import Strategy
from .context import StrategyContext

class LazyStrategy(Strategy):
    """Lazy：批处理层不做任何回填；迁移在读路径发生（读修复/回写）。"""
    name = "lazy"

    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        return []
