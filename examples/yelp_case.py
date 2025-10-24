# -*- coding: utf-8 -*-
"""
yelp_case.py
单窗口迁移-回放-聚合脚本（指标埋点增强版，带批量回放）
用法示例：
  # 只迁 100 条，快速跑通（以 BCO 为例）
  python yelp_case.py migrate   --strategy BCO --suffix bco --budget 100 --batch 100 --window 20251024T1000 --ps_size 100
  python yelp_case.py replay    --suffix bco  --window 20251024T1000 --qps 30 --duration 10 --seed 42
  python yelp_case.py aggregate --suffix bco  --window 20251024T1000 --p95_target 100

  # 一键跑所有策略各 1 窗（每个 100 条，极简演示）
  python yelp_case.py run_all --budget 100 --batch 100 --qps 30 --duration 10 --seed 42 --ps_size 100 --p95_target 100
"""
import os, sys, re, time, math, uuid, random, argparse, datetime as dt
from typing import Dict, Any, List, Optional
from bson import ObjectId, BSON
from pymongo import MongoClient, UpdateOne, ASCENDING

# --- 修正 sys.path：examples/ 下运行也能 import strategies 包 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 你的策略包（确保 strategies/__init__.py 导出类） ---
from strategies import (
    StrategyContext, Strategy,
    EagerStrategy, LazyStrategy, IncrementalStrategy, PredictiveStrategy,
    RequirementAdaptiveStrategy,BcoStrategy, LiteratureMixStrategy, TwoPhaseMABSelector
)

# ================== Mongo 配置 ==================
MONGO_URI  = os.getenv("MONGO_URI", "mongodb://duwendi:duwendi@223.223.185.189:12130/?authSource=admin")
DB_NAME    = "yelp_case"
SRC_COLL   = "reviews_S0"
DST_BASE   = "reviews_S7"

METRICS_COLL  = "migration_metrics"
BATCHLOG_COLL = "migration_batches"
READLOGS_COLL = "read_logs"
QUAR_COLL     = "migration_quarantine"

# ================== 工具函数 ==================
def to_number(x, default=0.0):
    if isinstance(x, (int, float)): return float(x)
    try: return float(x)
    except: return float(default)

def bson_size(doc: Dict[str, Any]) -> int:
    try: return len(BSON.encode(doc))
    except: return 0

def extract_title(text: str, maxlen: int = 80) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    first = text.strip().splitlines()[0]
    parts = re.split(r"(?<=[.!?。！？])\s+", first, maxsplit=1)
    title = parts[0] if parts else first
    return title[:maxlen].strip()

TAG_PAT = re.compile(r"(?:#|\[)([A-Za-z0-9_]+)(?:\])")
def extract_tags_csv(text: str) -> str:
    if not isinstance(text, str): return ""
    tags = TAG_PAT.findall(text.lower())
    return ",".join(sorted(set(tags))) if tags else ""

def summarize_reactions(useful, funny, cool) -> str:
    def n(v):
        try: return int(v)
        except: return 0
    return f"u={n(useful)},f={n(funny)},c={n(cool)}"

def ensure_target_indexes(coll):
    try: coll.create_index([("review_id", ASCENDING)], unique=True, name="ux_review_id")
    except: pass

# ================== S0 -> S7 转换（按你给的 schema 变更） ==================
def transform_s0_to_s7(d: Dict[str, Any]) -> Dict[str, Any]:
    review_id   = d.get("review_id", "")
    user_id     = d.get("user_id", "")
    business_id = d.get("business_id", "")

    rating = to_number(d.get("stars", 0))
    date   = d.get("date", "");  date = date if isinstance(date, str) else str(date)

    text  = d.get("text", "") if isinstance(d.get("text"), str) else ""
    title = extract_title(text)
    body  = text

    useful = d.get("useful", 0); funny = d.get("funny", 0); cool = d.get("cool", 0)
    reactions = {
        "useful": int(useful) if isinstance(useful, (int,float)) else 0,
        "funny":  int(funny)  if isinstance(funny,  (int,float)) else 0,
        "cool":   int(cool)   if isinstance(cool,   (int,float)) else 0,
        "summary": summarize_reactions(useful, funny, cool),
        "tags_csv": extract_tags_csv(text),
    }

    return {
        "review_id": review_id,
        "user_id": user_id,
        "business_id": business_id,
        "rating": rating,
        "date": date,
        "title": title,
        "body": body,
        "reactions": reactions,
        "embedded_business": {},
        "rating_detail": {},
        "rating_avg": rating,
    }

# ================== 批处理迁移（批次日志 + 错误分类） ==================
def migrate_batch_selected(src_docs, dst, quarantine, window_id: str) -> Dict[str, Any]:
    import pymongo
    t0 = time.perf_counter(); cpu0 = time.process_time()
    ops=[]; docs_ok=docs_failed=0; br=bw=0
    dup_key_err = validate_err = cast_err = other_err = retry_count = 0
    matched_count = modified_count = upserted_count = 0

    for d in src_docs:
        try:
            out = transform_s0_to_s7(d)
            br += bson_size(d); bw += bson_size(out)
            ops.append(UpdateOne({"review_id": out["review_id"]}, {"$set": out}, upsert=True))
            docs_ok += 1
        except Exception as e:
            msg = str(e)
            if "duplicate key" in msg: dup_key_err += 1
            elif "validation" in msg:  validate_err += 1
            elif "cast" in msg:        cast_err += 1
            else:                      other_err += 1
            quarantine.insert_one({"window_id": window_id, "src_id": d.get("_id"), "error": msg, "ts": time.time()})
            docs_failed += 1

    result = None
    if ops:
        try:
            result = dst.bulk_write(ops, ordered=False)
        except pymongo.errors.BulkWriteError:
            retry_count += 1
            for op in ops:
                try:
                    r = dst.bulk_write([op], ordered=True)
                    if r:
                        matched_count   += getattr(r, "matched_count", 0)
                        modified_count  += getattr(r, "modified_count", 0)
                        upserted_count  += len(getattr(r, "upserted_ids", {}) or {})
                except Exception as ee:
                    msg = str(ee)
                    if "duplicate key" in msg: dup_key_err += 1
                    elif "validation" in msg:  validate_err += 1
                    elif "cast" in msg:        cast_err += 1
                    else:                      other_err += 1
                    quarantine.insert_one({"window_id": window_id, "upsert_filter": getattr(op, "_filter", None), "error": msg, "ts": time.time()})
                    docs_failed += 1
        except Exception:
            retry_count += 1
            for op in ops:
                try:
                    r = dst.bulk_write([op], ordered=True)
                    if r:
                        matched_count   += getattr(r, "matched_count", 0)
                        modified_count  += getattr(r, "modified_count", 0)
                        upserted_count  += len(getattr(r, "upserted_ids", {}) or {})
                except Exception as e3:
                    msg = str(e3)
                    if "duplicate key" in msg: dup_key_err += 1
                    elif "validation" in msg:  validate_err += 1
                    elif "cast" in msg:        cast_err += 1
                    else:                      other_err += 1
                    quarantine.insert_one({"window_id": window_id, "upsert_filter": getattr(op, "_filter", None), "error": msg, "ts": time.time()})
                    docs_failed += 1
        else:
            if result:
                matched_count   += getattr(result, "matched_count", 0)
                modified_count  += getattr(result, "modified_count", 0)
                upserted_count  += len(getattr(result, "upserted_ids", {}) or {})

    cpu_ms = (time.process_time()-cpu0)*1000.0
    el_ms  = (time.perf_counter()-t0)*1000.0
    return {
        "batch_id": str(uuid.uuid4())[:8],
        "docs_ok": docs_ok, "docs_failed": docs_failed,
        "bytes_read": br, "bytes_write": bw,
        "cpu_ms": cpu_ms, "elapsed_ms": el_ms,
        "matched_count": matched_count, "modified_count": modified_count, "upserted_count": upserted_count,
        "dup_key_err": dup_key_err, "validate_err": validate_err, "cast_err": cast_err, "other_err": other_err,
        "retry_count": retry_count
    }

# ================== 读负载回放（批量判断 + 批量写 logs） ==================
def zipf_indices(n: int, size: int, a: float = 1.2) -> List[int]:
    try:
        import numpy as np
        arr = np.random.zipf(a=a, size=size)
        arr = (arr % n)
        return arr.tolist()
    except Exception:
        xs=[]
        for _ in range(size):
            r=random.random()
            idx = int((r ** (-1.0/a))) % max(1,n)
            xs.append(idx)
        return xs

def replay_reads(cli: MongoClient, dst_name: str, dst_suffix: str, window_id: str,
                 qps: int, duration_sec: int, zipf_a: float, seed: int, run_id: str):
    db  = cli[DB_NAME]
    src = db[SRC_COLL]
    dst = db[dst_name]
    logs= db[READLOGS_COLL]
    random.seed(seed)

    pool = list(src.find({}, {"review_id":1, "text":1}).limit(200_000))
    if not pool: return
    N = len(pool); total_req = qps * duration_sec; idxs = zipf_indices(N, total_req, a=zipf_a)

    base_mean, base_jit = 6.0, 2.0
    BULK = 1000
    buf = []
    pending = []

    for k in idxs:
        item = pool[k]; rid=item.get("review_id"); text=item.get("text","")
        pending.append((rid, text))
        if len(pending) >= BULK:
            rids = [r for r,_ in pending]
            migrated = set(x["review_id"] for x in dst.find({"review_id":{"$in": rids}}, {"review_id":1}))
            for rid, text in pending:
                is_old = rid not in migrated
                L = len(text)
                if L < 200: dL = 4 + random.random()*4
                elif L < 800: dL = 8 + random.random()*7
                else: dL = 15 + random.random()*15
                udf_cpu_ms = dL*0.6 if is_old else 0.0
                io_blocks  = dL*0.4 if is_old else 0.0
                lat = (base_mean + (random.random()-0.5)*base_jit) + (dL if is_old else 0.0)
                buf.append({
                    "run_id": run_id, "window_id": window_id,
                    "strategy_name": dst_suffix, "review_id": rid,
                    "lat_ms_total": float(lat),
                    "udf_cpu_ms": float(udf_cpu_ms),
                    "io_blocks_transform": float(io_blocks),
                    "is_old": bool(is_old),
                    "ts": time.time()
                })
            if buf:
                logs.insert_many(buf, ordered=False)
                buf.clear()
            pending.clear()

    if pending:
        rids = [r for r,_ in pending]
        migrated = set(x["review_id"] for x in dst.find({"review_id":{"$in": rids}}, {"review_id":1}))
        for rid, text in pending:
            is_old = rid not in migrated
            L = len(text)
            if L < 200: dL = 4 + random.random()*4
            elif L < 800: dL = 8 + random.random()*7
            else: dL = 15 + random.random()*15
            udf_cpu_ms = dL*0.6 if is_old else 0.0
            io_blocks  = dL*0.4 if is_old else 0.0
            lat = (base_mean + (random.random()-0.5)*base_jit) + (dL if is_old else 0.0)
            buf.append({
                "run_id": run_id, "window_id": window_id,
                "strategy_name": dst_suffix, "review_id": rid,
                "lat_ms_total": float(lat),
                "udf_cpu_ms": float(udf_cpu_ms),
                "io_blocks_transform": float(io_blocks),
                "is_old": bool(is_old),
                "ts": time.time()
            })
        if buf:
            logs.insert_many(buf, ordered=False)
            buf.clear()

# ================== 聚合指标（更多分位数 + 直方图 + P/R） ==================
def aggregate_window_metrics(cli: MongoClient, dst_suffix: str, window_id: str,
                             p95_target_ms: float = 100.0, beta_f: float = 1.0,
                             selected_ids: Optional[List[Any]] = None):
    db = cli[DB_NAME]
    dst = db[f"{DST_BASE}_{dst_suffix}" if dst_suffix else DST_BASE]
    metrics = db[METRICS_COLL]
    logs = db[READLOGS_COLL]

    cur = logs.find(
        {"window_id": window_id, "strategy_name": dst_suffix},
        {"lat_ms_total":1,"udf_cpu_ms":1,"io_blocks_transform":1,"is_old":1,"review_id":1,"_id":0}
    )

    lats=[]; on_read_cost=0.0; total=0; old_hits=0
    buckets = [2,5,10,20,50,100,200,500]
    hist = [0]*len(buckets)
    accessed_ids = set()

    for x in cur:
        lat = float(x.get("lat_ms_total",0.0)); lats.append(lat)
        on_read_cost += float(x.get("udf_cpu_ms",0.0)) + float(x.get("io_blocks_transform",0.0))
        total += 1
        if x.get("is_old"): old_hits += 1
        rid = x.get("review_id");
        if rid: accessed_ids.add(rid)
        # 落桶
        for i, le in enumerate(buckets):
            if lat <= le: hist[i]+=1; break

    def pctl(arr, q):
        if not arr: return None
        arr = sorted(arr)
        k = int(math.ceil(q*len(arr))) - 1
        k = max(0, min(k, len(arr)-1))
        return arr[k]
    p50 = pctl(lats, 0.50); p90 = pctl(lats, 0.90); p95 = pctl(lats, 0.95); p99 = pctl(lats, 0.99); p999 = pctl(lats, 0.999)
    hit_rate = (old_hits/total) if total else 0.0

    # Precision / Recall / Fβ （需要传入 selected_ids）
    precision = recall = fbeta = None
    if selected_ids is not None:
        sel = set(str(x) for x in selected_ids)
        acc = set(str(x) for x in accessed_ids)
        tp  = len(sel & acc)
        fp  = max(0, len(sel) - tp)
        fn  = max(0, len(acc) - tp)
        precision = (tp / (tp + fp)) if (tp+fp)>0 else None
        recall    = (tp / (tp + fn)) if (tp+fn)>0 else None
        if precision is not None and recall is not None and (precision+recall)>0:
            beta2 = beta_f * beta_f
            fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall)

    metrics.update_one(
        {"window_id": window_id, "coll": f"{DB_NAME}.{dst.name}"},
        {"$set": {
            "p50_ms": p50, "p90_ms": p90, "p95_ms": p95, "p99_ms": p99, "p999_ms": p999,
            "p95_target_ms": p95_target_ms,
            "on_read_cost": on_read_cost,
            "old_hit_rate": hit_rate,
            "lat_hist": [{"le":buckets[i], "cnt": hist[i]} for i in range(len(buckets))],
            "precision": precision, "recall": recall, "fbeta": fbeta, "beta_f": beta_f
        }}, upsert=True
    )
    print(f"[aggregate {dst.name} {window_id}] p95={p95} on_read={on_read_cost:.2f} hit={hit_rate:.3f} P/R={precision}/{recall}")

# ================== 迁移窗口（选 id → 迁移 → 写 metrics） ==================
STRAT_NAME_MAP = {
    "EagerStrategy": EagerStrategy,
    "LazyStrategy": LazyStrategy,
    "IncrementalStrategy": IncrementalStrategy,
    "PredictiveStrategy": PredictiveStrategy,
    "RequirementAdaptiveStrategy": RequirementAdaptiveStrategy,
    "LiteratureMixStrategy": LiteratureMixStrategy,
    "TwoPhaseMABSelector": TwoPhaseMABSelector,
    "BcoStrategy": BcoStrategy,          # ← 新增

    # 简写/别名
    "Eager": EagerStrategy,
    "Lazy": LazyStrategy,
    "Incremental": IncrementalStrategy,
    "Predictive": PredictiveStrategy,
    "RequirementAdaptive": RequirementAdaptiveStrategy,
    "LiteratureMix": LiteratureMixStrategy,
    "TwoPhaseMAB": TwoPhaseMABSelector,
    "BCO": BcoStrategy,                  # ← 新增
    "Bco": BcoStrategy,                  # ← 新增
}

def migrate_once(cli: MongoClient, strategy_name: str, dst_suffix: str,
                 window_id: str, run_id: str, window_budget: int, batch_size: int,
                 ps_size: int, p95_target_ms: float):
    db  = cli[DB_NAME]; src = db[SRC_COLL]
    dst_name = f"{DST_BASE}_{dst_suffix}" if dst_suffix else DST_BASE
    dst = db[dst_name]
    metrics = db[METRICS_COLL]; batches = db[BATCHLOG_COLL]; quarantine = db[QUAR_COLL]
    ensure_target_indexes(dst)

    # 1) 选 id（硬截断到 budget）
    Strat = STRAT_NAME_MAP[strategy_name]
    ctx = StrategyContext(
        coll=src, version_field="_sv", target_version="S7",
        window_budget=window_budget, access_stats_coll="access_stats",
        p95_ms=120.0, p95_target_ms=p95_target_ms,
        on_release_cost=0.0, on_read_cost=0.0, cost_window_budget=5.0,
        ps_size=min(ps_size, window_budget)
    )
    strat = Strat()
    raw_ids = strat.select_ids(ctx) or []
    seen=set(); dedup=[]
    for x in raw_ids:
        if x in seen: continue
        seen.add(x); dedup.append(x)
    picked_ids = dedup[:window_budget]

    # 2) 迁移（记录批次日志）
    t0 = time.perf_counter()
    docs_ok=docs_failed=0; bytes_r=bytes_w=0; cpu_ms_total=0.0
    for i in range(0, len(picked_ids), batch_size):
        chunk = picked_ids[i:i+batch_size]
        # 兼容 _id:ObjectId / review_id:str
        obj_ids = []
        for x in chunk:
            if isinstance(x, ObjectId): obj_ids.append(x)
            elif isinstance(x, str) and len(x)==24:
                try: obj_ids.append(ObjectId(x))
                except: pass
        if obj_ids:
            src_docs = list(src.find({"_id": {"$in": obj_ids}}))
        else:
            src_docs = list(src.find({"review_id": {"$in": chunk}}))

        rep = migrate_batch_selected(src_docs, dst, quarantine, window_id)
        batches.insert_one({
            "run_id": run_id, "window_id": window_id, "strategy_name": dst_suffix,
            "from_coll": f"{DB_NAME}.{SRC_COLL}", "coll": f"{DB_NAME}.{dst_name}", **rep, "picked_ids_count": len(chunk), "ts": time.time()
        })
        docs_ok     += rep["docs_ok"];    docs_failed += rep["docs_failed"]
        bytes_r     += rep["bytes_read"]; bytes_w     += rep["bytes_write"]
        cpu_ms_total+= rep["cpu_ms"]

    t1 = time.perf_counter()
    total_src = src.estimated_document_count()
    dst_count = dst.estimated_document_count()
    debt_ratio = (max(0, total_src-dst_count)/total_src) if total_src else 0.0

    io_mb = (bytes_r + bytes_w) / 1024.0 / 1024.0
    on_release_cost = cpu_ms_total + io_mb

    # 聚合批次耗时、错误像
    agg = list(batches.aggregate([
        {"$match": {"run_id": run_id, "window_id": window_id, "strategy_name": dst_suffix}},
        {"$group": {
            "_id": None,
            "elapsed_ms": {"$sum": {"$ifNull": ["$elapsed_ms", 0]}},
            "dup_key_err": {"$sum": {"$ifNull": ["$dup_key_err", 0]}},
            "validate_err": {"$sum": {"$ifNull": ["$validate_err", 0]}},
            "cast_err": {"$sum": {"$ifNull": ["$cast_err", 0]}},
            "other_err": {"$sum": {"$ifNull": ["$other_err", 0]}},
            "retry_count": {"$sum": {"$ifNull": ["$retry_count", 0]}}
        }}
    ]))
    agg0 = agg[0] if agg else {}
    total_elapsed_ms = float(agg0.get("elapsed_ms", 0.0))
    throughput = (docs_ok / max(1.0, total_elapsed_ms/1000.0))

    params_snapshot = {
        "window_budget": window_budget,
        "ps_size": min(ps_size, window_budget),
        "p95_target_ms": p95_target_ms,
        "cost_window_budget": 5.0
    }

    metrics.update_one(
        {"run_id": run_id, "window_id": window_id, "coll": f"{DB_NAME}.{dst_name}"},
        {"$set": {
            "run_id": run_id, "window_id": window_id, "coll": f"{DB_NAME}.{dst_name}",
            "strategy_name": dst_suffix, "params": params_snapshot,
            "ts_start": t0, "ts_end": t1,
            "picked_ids_count": len(picked_ids),
            "docs_ok": docs_ok, "docs_failed": docs_failed, "docs_skipped": 0,
            "bytes_read": bytes_r, "bytes_write": bytes_w, "io_mb": io_mb,
            "cpu_ms": cpu_ms_total, "on_release_cost": on_release_cost,
            "throughput_docs_per_s": throughput,
            "debt_ratio": debt_ratio,
            "dup_key_err": agg0.get("dup_key_err", 0), "validate_err": agg0.get("validate_err", 0),
            "cast_err": agg0.get("cast_err", 0), "other_err": agg0.get("other_err", 0),
            "retry_count": agg0.get("retry_count", 0),
            # 读期指标聚合后再补
            "p50_ms": None, "p90_ms": None, "p95_ms": None, "p99_ms": None, "p999_ms": None,
            "on_read_cost": None, "old_hit_rate": None, "lat_hist": None,
            "precision": None, "recall": None, "fbeta": None, "beta_f": 1.0
        }}, upsert=True
    )
    print(f"[migrate {dst_name} {dst_suffix} {window_id}] want={len(raw_ids)} cap={window_budget} picked={len(picked_ids)} ok={docs_ok} fail={docs_failed} debt={debt_ratio:.4f}")
    return picked_ids  # 用于 P/R

# ================== CLI ==================
def main():
    ap = argparse.ArgumentParser("Yelp case: migrate/replay/aggregate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_m = sub.add_parser("migrate")
    ap_m.add_argument("--strategy", required=True, help="Eager/Lazy/Incremental/Predictive/RequirementAdaptive/LiteratureMix/TwoPhaseMAB/BCO")
    ap_m.add_argument("--suffix",   required=True, help="目标集合后缀，如 bco/predictive")
    ap_m.add_argument("--window",   required=True, help="窗口号，如 20251024T1000")
    ap_m.add_argument("--budget",   type=int, default=1500)
    ap_m.add_argument("--batch",    type=int, default=500)
    ap_m.add_argument("--ps_size",  type=int, default=2000)
    ap_m.add_argument("--p95_target", type=float, default=100.0)

    ap_r = sub.add_parser("replay")
    ap_r.add_argument("--suffix",   required=True)
    ap_r.add_argument("--window",   required=True)
    ap_r.add_argument("--qps",      type=int, default=400)
    ap_r.add_argument("--duration", type=int, default=90)
    ap_r.add_argument("--zipf",     type=float, default=1.2)
    ap_r.add_argument("--seed",     type=int, default=42)

    ap_a = sub.add_parser("aggregate")
    ap_a.add_argument("--suffix",   required=True)
    ap_a.add_argument("--window",   required=True)
    ap_a.add_argument("--p95_target", type=float, default=100.0)

    ap_all = sub.add_parser("run_all")
    ap_all.add_argument("--budget",   type=int, default=100)
    ap_all.add_argument("--batch",    type=int, default=100)
    ap_all.add_argument("--qps",      type=int, default=30)
    ap_all.add_argument("--duration", type=int, default=10)
    ap_all.add_argument("--seed",     type=int, default=42)
    ap_all.add_argument("--ps_size",  type=int, default=100)
    ap_all.add_argument("--p95_target", type=float, default=100.0)

    args = ap.parse_args()
    cli = MongoClient(MONGO_URI)

    if args.cmd == "migrate":
        run_id = args.window.split("-")[0] if "-" in args.window else args.window
        migrate_once(cli, args.strategy, args.suffix, args.window, run_id, args.budget, args.batch, args.ps_size, args.p95_target)

    elif args.cmd == "replay":
        run_id = args.window.split("-")[0] if "-" in args.window else args.window
        dst_name = f"{DST_BASE}_{args.suffix}"
        # 让不同策略的随机流不同（更容易拉开 p95）
        stable_offset = (abs(hash(args.suffix)) % 997)
        replay_reads(cli, dst_name, args.suffix, args.window,
                     qps=args.qps, duration_sec=args.duration,
                     zipf_a=args.zipf, seed=args.seed + stable_offset,
                     run_id=run_id)
        print(f"[replay {dst_name} {args.window}] done")

    elif args.cmd == "aggregate":
        run_id = args.window.split("-")[0] if "-" in args.window else args.window
        aggregate_window_metrics(cli, dst_suffix=args.suffix, window_id=args.window, p95_target_ms=args.p95_target, beta_f=1.0, selected_ids=None)

    elif args.cmd == "run_all":
        run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        window_id = run_id + "-w0"
        roster = [
            ("Eager", "eager"),
            ("Lazy", "lazy"),
            ("Incremental", "incremental"),
            ("Predictive", "predictive"),
            ("RequirementAdaptive", "requirementadaptive"),
            ("LiteratureMix", "literaturemix"),
            ("TwoPhaseMAB", "twophasemab"),
            ("BCO", "bco"),
        ]
        for strat, suffix in roster:
            picked = migrate_once(cli, strat, suffix, window_id, run_id,
                                  window_budget=args.budget, batch_size=args.batch,
                                  ps_size=args.ps_size, p95_target_ms=args.p95_target)
            stable_offset = (abs(hash(suffix)) % 997)
            replay_reads(cli, f"{DST_BASE}_{suffix}", suffix, window_id,
                         qps=args.qps, duration_sec=args.duration,
                         zipf_a=1.2, seed=args.seed + stable_offset, run_id=run_id)
            aggregate_window_metrics(cli, dst_suffix=suffix, window_id=window_id,
                                     p95_target_ms=args.p95_target, beta_f=1.0, selected_ids=picked)
        print(f"== DONE run_id={run_id} ==")

if __name__ == "__main__":
    main()
