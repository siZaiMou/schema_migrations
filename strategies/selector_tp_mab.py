from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
import random
from .context import StrategyContext
from .base import Strategy
from .eager import EagerStrategy
from .lazy import LazyStrategy
from .incremental import IncrementalStrategy
from .predictive import PredictiveStrategy
from .requirement_adaptive import RequirementAdaptiveStrategy
from .bco import BcoStrategy

class TwoPhaseMABSelector:
    """
    两阶段多臂赌博选择器（分配窗口预算 -> 让各策略各自挑 _id -> 合并去重）
    Phase-1: 先验配重（基于 SLA 压力、Π 复杂度、预算约束的启发式）
    Phase-2: 在线 MAB（Thompson/Soft-UCB）依据观测回报调权，内置 SLA 安全闸
    使用方法：
      selector = TwoPhaseMABSelector()
      ids = selector.select_ids(ctx)
      selector.update_rewards(R_t)   # 每个窗口结束后喂回回报
    """

    def __init__(self,
                 arms: Dict[str, Strategy] = None,
                 # 回报权重（与我们前面定义的综合指标一致）
                 w_latency: float = 0.4, w_cost: float = 0.2, w_fbeta: float = 0.3, w_debt: float = 0.1,
                 beta_f: float = 1.5,
                 # 探索/利用参数
                 ucb_temp: float = 0.5,   # soft-UCB 温度
                 # SLA 安全闸
                 p95_guard_ratio: float = 0.8,   # 超过 0.8×阈值 → 提升激进臂权重
                 max_step: float = 0.25          # 单窗口内权重最多变化幅度
                 ):
        # 策略臂集合（可扩展/替换）
        self.arms: Dict[str, Strategy] = arms or {
            "bco": BcoStrategy(),
            "predictive": PredictiveStrategy(),
            "incremental": IncrementalStrategy(),
            # 也可接入 eager/lazy，但通常作为边界臂
            # "eager": EagerStrategy(),
            # "lazy": LazyStrategy(),
            "req_adaptive": RequirementAdaptiveStrategy(),
        }
        # 权重（概率）初始化
        self.weights: Dict[str, float] = {k: 1.0/len(self.arms) for k in self.arms}
        # 统计（用于 UCB / TS 的经验回报）
        self.history: Dict[str, List[float]] = {k: [] for k in self.arms}

        self.w_latency, self.w_cost, self.w_fbeta, self.w_debt = w_latency, w_cost, w_fbeta, w_debt
        self.beta_f = beta_f
        self.ucb_temp = ucb_temp
        self.p95_guard_ratio = p95_guard_ratio
        self.max_step = max_step

    # ========== 回报函数（把指标规一到 0~1） ==========
    def _reward(self, metrics: Dict[str, float], ctx: StrategyContext) -> float:
        # 需要：P95、总成本、Fβ、债务比例
        L = metrics.get("p95_ms", None)
        C = metrics.get("cost_total", None)
        Prec = metrics.get("precision", None)
        Rec  = metrics.get("recall", None)
        Debt = metrics.get("debt_ratio", None)

        # 目标上限
        Lmax = ctx.p95_target_ms or (L if L else 1.0)
        Cmax = ctx.cost_window_budget or (C if C else 1.0)

        part_L = (1.0 - min(1.0, (L or Lmax)/Lmax))
        part_C = (1.0 - min(1.0, (C or Cmax)/Cmax))
        # Fβ（缺失时退化）
        if Prec is None or Rec is None:
            part_F = 0.0
        else:
            beta2 = self.beta_f**2
            denom = beta2*Prec + Rec
            part_F = (1+beta2)*Prec*Rec/denom if denom>0 else 0.0
        part_D = 1.0 - min(1.0, Debt) if Debt is not None else 0.0

        R = self.w_latency*part_L + self.w_cost*part_C + self.w_fbeta*part_F + self.w_debt*part_D
        return max(0.0, min(1.0, R))

    # ========== Phase-1：先验（启发式权重） ==========
    def _prior_weights(self, ctx: StrategyContext) -> Dict[str, float]:
        w = {k: 1e-6 for k in self.arms}  # 防止 0
        # SLA 压力
        sla_pressure = 0.0
        if ctx.p95_ms is not None and ctx.p95_target_ms:
            sla_pressure = min(1.0, ctx.p95_ms/ctx.p95_target_ms)
        # 启发式：压力高 → bco/req_adaptive/predictive 增权；其余走 incremental
        for k in self.arms:
            if k == "bco":
                w[k] += 0.3 + 0.3*sla_pressure
            elif k == "req_adaptive":
                w[k] += 0.25*sla_pressure + 0.05
            elif k == "predictive":
                w[k] += 0.2*sla_pressure + 0.1
            else:  # incremental/eager/lazy
                w[k] += 0.35*(1.0 - 0.5*sla_pressure)
        # 归一
        s = sum(w.values())
        return {k: v/s for k,v in w.items()}

    # ========== Phase-2：Soft-UCB 调权 + SLA 安全闸 ==========
    def _online_adjust(self, prior: Dict[str,float], ctx: StrategyContext) -> Dict[str,float]:
        # 计算每臂平均回报与置信项（UCB）
        means = {k: (sum(self.history[k])/len(self.history[k]) if self.history[k] else 0.5) for k in self.arms}
        counts = {k: max(1, len(self.history[k])) for k in self.arms}
        totalN = sum(counts.values())

        scores = {}
        for k in self.arms:
            bonus = math.sqrt(2*math.log(max(2,totalN))/counts[k])
            scores[k] = means[k] + self.ucb_temp * bonus

        # Softmax 到 [0,1]
        mx = max(scores.values()) if scores else 1.0
        exps = {k: math.exp( (scores[k]-mx) / max(1e-6, self.ucb_temp) ) for k in scores}
        s = sum(exps.values()) or 1.0
        online = {k: exps[k]/s for k in scores}

        # 融合先验：w = normalize( prior * online )
        fused = {k: prior.get(k,0.0) * online.get(k,0.0) for k in self.arms}
        fs = sum(fused.values()) or 1.0
        fused = {k: v/fs for k,v in fused.items()}

        # SLA 安全闸：P95 逼近阈值时，提升 bco/req_adaptive 权重、降低 lazy/incremental 一点点
        if ctx.p95_ms is not None and ctx.p95_target_ms and (ctx.p95_ms >= self.p95_guard_ratio*ctx.p95_target_ms):
            for k in fused:
                if k in ("bco","req_adaptive"):
                    fused[k] = min(1.0, fused[k] + 0.1)
                elif k in ("lazy",):
                    fused[k] = max(0.0, fused[k] - 0.1)
            # 归一
            s2 = sum(fused.values()) or 1.0
            fused = {k: v/s2 for k,v in fused.items()}

        # 平滑：限制单窗口最大变化
        out = {}
        for k in fused:
            prev = self.weights.get(k, 0.0)
            delta = max(-self.max_step, min(self.max_step, fused[k] - prev))
            out[k] = max(0.0, prev + delta)
        s3 = sum(out.values()) or 1.0
        return {k: v/s3 for k,v in out.items()}

    # ========== 主入口 ==========
    def select_ids(self, ctx: StrategyContext) -> List[Any]:
        # Phase-1
        prior = self._prior_weights(ctx)
        # Phase-2
        self.weights = self._online_adjust(prior, ctx)

        # 切预算并让各臂各自挑 _id
        B = ctx.window_budget
        ids: List[Any] = []
        alloc: Dict[str,int] = {}
        acc = 0
        for k,w in self.weights.items():
            n = int(B * w)
            alloc[k] = n
            acc += n
        # 把剩余的补给最高权重的臂
        if acc < B:
            k_max = max(self.weights, key=self.weights.get)
            alloc[k_max] += (B - acc)

        for k, n in alloc.items():
            if n <= 0: continue
            subctx = StrategyContext(**{**ctx.__dict__, "window_budget": n})
            ids.extend(self.arms[k].select_ids(subctx))

        # 去重 & 截断
        seen, uniq = set(), []
        for _id in ids:
            if _id not in seen:
                seen.add(_id)
                uniq.append(_id)
            if len(uniq) >= B:
                break
        return uniq

    # ========== 反馈更新：把一个窗口的指标回传成回报 ==========
    def update_rewards(self, metrics: Dict[str, float], ctx: StrategyContext):
        R = self._reward(metrics, ctx)  # 0~1
        # 简单地把总回报均分给当期出场的臂（也可按各臂贡献量分配）
        for k,w in self.weights.items():
            if w > 0:
                self.history[k].append(R)
