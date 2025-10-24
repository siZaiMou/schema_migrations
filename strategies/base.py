from __future__ import annotations
from typing import Any, List
from .context import StrategyContext

class Strategy:
    """策略接口：决定‘这一窗口迁哪批 _id’。"""
    name: str = "base"

    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        raise NotImplementedError

    def adjust_prediction_set(self, ctx: StrategyContext):
        """部分策略（如自适应）可动态调整 ps_size；默认不调整。"""
        return None
