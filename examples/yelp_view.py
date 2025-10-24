# -*- coding: utf-8 -*-
"""
yelp_view_multi.py
按主题/策略输出多张 PNG：概览图、分位延迟、总成本、失败画像、吞吐、P/R/Fβ、每策略直方图/箱线图等
用法：
  python yelp_view.py --run_id 20251024T025316 --outdir figs_20251024T025316 --boxplot_sample 2000
依赖：pymongo pandas numpy matplotlib
"""
import os, argparse, math, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://duwendi:duwendi@223.223.185.189:12130/?authSource=admin")
DB_NAME   = "yelp_case"
METRICS   = "migration_metrics"
BATCHES   = "migration_batches"
READLOGS  = "read_logs"

ORDER = ["eager","lazy","incremental","predictive","requirementadaptive",
         "literaturemix","twophasemab","bco","mab"]
SEL_STRATEGIES = {"predictive","requirementadaptive","bco","twophasemab"}

def to_float_series(s, default=0.0):
    if s is None:
        return pd.Series(dtype=float)
    try:
        return pd.to_numeric(s, errors="coerce").fillna(default)
    except Exception:
        return pd.Series([default]*len(s))

def fetch_data(run_id: str):
    cli = MongoClient(MONGO_URI)
    db  = cli[DB_NAME]
    m = list(db[METRICS].find({"run_id": run_id}, {"_id":0}))
    b = list(db[BATCHES].find({"run_id": run_id}, {"_id":0}))
    l = list(db[READLOGS].find({"run_id": run_id}, {"_id":0, "strategy_name":1, "window_id":1, "lat_ms_total":1}))
    return pd.DataFrame(m), pd.DataFrame(b), pd.DataFrame(l)

def order_by_strategy(df, col="strategy_name"):
    if df is None or df.empty or col not in df:
        return df
    s = df[col].astype(str).str.lower()
    cat = pd.Categorical(s, categories=ORDER, ordered=True)
    out = df.copy()
    out[col] = cat
    return out.sort_values(col)

def xticks(ax, labels):
    xs = np.arange(len(labels))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    return xs

def ensure_outdir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def savefig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Saved:", path)

def main():
    ap = argparse.ArgumentParser("Multi-figure visualization for one run_id")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--outdir", default=None, help="输出目录（默认 figs_<run_id>）")
    ap.add_argument("--csv_prefix", default=None)
    ap.add_argument("--boxplot_sample", type=int, default=2000)
    args = ap.parse_args()

    outdir = args.outdir or f"figs_{args.run_id}"
    ensure_outdir(outdir)

    mdf, bdf, ldf = fetch_data(args.run_id)
    if mdf.empty:
        print("No metrics for run_id", args.run_id); return

    # 统一小写
    mdf["strategy_name"] = mdf["strategy_name"].astype(str).str.lower()
    if not bdf.empty: bdf["strategy_name"] = bdf["strategy_name"].astype(str).str.lower()
    if not ldf.empty: ldf["strategy_name"] = ldf["strategy_name"].astype(str).str.lower()

    # 批次聚合（on-release 拆分 & 错误）
    if not bdf.empty:
        agg = bdf.groupby("strategy_name", as_index=False, observed=False).agg({
            "cpu_ms":"sum" if "cpu_ms" in bdf.columns else "size",
            "bytes_read":"sum" if "bytes_read" in bdf.columns else "size",
            "bytes_write":"sum" if "bytes_write" in bdf.columns else "size",
            "docs_ok":"sum" if "docs_ok" in bdf.columns else "size",
            "docs_failed":"sum" if "docs_failed" in bdf.columns else "size",
            "elapsed_ms":"sum" if "elapsed_ms" in bdf.columns else "size",
            "dup_key_err":"sum" if "dup_key_err" in bdf.columns else "size",
            "validate_err":"sum" if "validate_err" in bdf.columns else "size",
            "cast_err":"sum" if "cast_err" in bdf.columns else "size",
            "other_err":"sum" if "other_err" in bdf.columns else "size",
            "retry_count":"sum" if "retry_count" in bdf.columns else "size",
        })
        for col in ["cpu_ms","bytes_read","bytes_write","docs_ok","docs_failed","elapsed_ms",
                    "dup_key_err","validate_err","cast_err","other_err","retry_count"]:
            if agg[col].dtype == np.int64 and col not in bdf.columns:
                agg[col] = 0
        agg["io_mb"] = (to_float_series(agg["bytes_read"]) + to_float_series(agg["bytes_write"])) / 1024.0 / 1024.0
    else:
        agg = pd.DataFrame(columns=["strategy_name","cpu_ms","io_mb","docs_ok","docs_failed","elapsed_ms",
                                    "dup_key_err","validate_err","cast_err","other_err","retry_count"])

    # 合并
    mdf = mdf.merge(agg[["strategy_name","cpu_ms","io_mb","docs_ok","docs_failed","elapsed_ms",
                         "dup_key_err","validate_err","cast_err","other_err","retry_count"]],
                    on="strategy_name", how="left")

    # 数值化 & 派生
    for col in ["p50_ms","p90_ms","p95_ms","p99_ms","p999_ms","on_read_cost","debt_ratio","old_hit_rate",
                "cpu_ms","io_mb","docs_ok","docs_failed","elapsed_ms","throughput_docs_per_s",
                "precision","recall","fbeta","on_release_cost"]:
        if col in mdf.columns: mdf[col] = to_float_series(mdf[col], default=0.0)
        else: mdf[col] = 0.0
    mdf["on_release_cost"] = mdf.apply(lambda r: r["on_release_cost"] if r["on_release_cost"]>0 else (r["cpu_ms"] + r["io_mb"]), axis=1)
    mdf["total_cost"] = mdf["on_release_cost"] + mdf["on_read_cost"]
    mdf["throughput_docs_per_s"] = mdf.apply(
        lambda r: r.get("throughput_docs_per_s", 0.0) if r.get("throughput_docs_per_s", None) not in (None, np.nan)
            else (r.get("docs_ok",0.0) / max(1.0, r.get("elapsed_ms",0.0)/1000.0)), axis=1
    )

    # 排序 + 标签
    mdf = order_by_strategy(mdf, "strategy_name")
    labels = mdf["strategy_name"].astype(str).tolist()
    xs = np.arange(len(labels))

    # ============ 1) 概览九宫格（和之前类似，但合成一张） ============
    fig = plt.figure(figsize=(16,12))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1,1,1.2])

    ax1 = fig.add_subplot(gs[0,0]); ax1.bar(xs, mdf["p95_ms"]); ax1.set_title("P95 (ms)"); ax1.set_xticks(xs); ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax2 = fig.add_subplot(gs[0,1]); ax2.bar(xs, mdf["p99_ms"]); ax2.set_title("P99 (ms)"); ax2.set_xticks(xs); ax2.set_xticklabels(labels, rotation=30, ha="right")

    ax3 = fig.add_subplot(gs[0,2])
    ax3.bar(xs, mdf["debt_ratio"], label="Debt Ratio"); ax3_t=ax3.twinx()
    ax3_t.plot(xs, mdf.get("old_hit_rate", pd.Series([0]*len(mdf))).values, "o--", label="Old Hit Rate")
    ax3.set_title("Debt vs Old-Hit"); ax3.set_xticks(xs); ax3.set_xticklabels(labels, rotation=30, ha="right")

    ax4 = fig.add_subplot(gs[1,0]); ax4.bar(xs, mdf["on_read_cost"]); ax4.set_title("On-read Cost"); ax4.set_xticks(xs); ax4.set_xticklabels(labels, rotation=30, ha="right")

    ax5 = fig.add_subplot(gs[1,1])
    bottom=np.zeros(len(labels))
    ax5.bar(xs, mdf["cpu_ms"], bottom=bottom, label="CPU ms"); bottom+=mdf["cpu_ms"].values
    ax5.bar(xs, mdf["io_mb"], bottom=bottom, label="IO MB")
    ax5.set_title("On-release Split"); ax5.legend(); ax5.set_xticks(xs); ax5.set_xticklabels(labels, rotation=30, ha="right")

    ax6 = fig.add_subplot(gs[1,2])
    bottom=np.zeros(len(labels))
    ax6.bar(xs, mdf["docs_ok"], bottom=bottom, label="OK"); bottom+=mdf["docs_ok"].values
    ax6.bar(xs, mdf["docs_failed"], bottom=bottom, label="Failed")
    ax6.set_title("Docs OK/Failed"); ax6.legend(); ax6.set_xticks(xs); ax6.set_xticklabels(labels, rotation=30, ha="right")

    ax7 = fig.add_subplot(gs[2,0]); ax7.bar(xs, mdf["throughput_docs_per_s"]); ax7.set_title("Throughput (docs/s)"); ax7.set_xticks(xs); ax7.set_xticklabels(labels, rotation=30, ha="right")

    # 直方图 / CDF（全日志）
    ax8 = fig.add_subplot(gs[2,1]); ax9 = fig.add_subplot(gs[2,2])
    # 直方
    # 读日志可能为空
    cli = MongoClient(MONGO_URI)
    ldf_all = pd.DataFrame(list(cli[DB_NAME][READLOGS].find({"run_id": args.run_id}, {"_id":0, "strategy_name":1, "lat_ms_total":1})))
    if not ldf_all.empty:
        arr = to_float_series(ldf_all["lat_ms_total"]).values
        if len(arr)>0:
            ax8.hist(arr, bins=30); ax8.set_title("Latency Histogram (All logs)")
        # CDF 按策略
        for s in labels:
            sub = ldf_all.loc[ldf_all["strategy_name"]==s, "lat_ms_total"]
            if sub.empty: continue
            vals = np.sort(to_float_series(sub).values)
            if len(vals)==0: continue
            y = np.linspace(0,1,len(vals))
            ax9.plot(vals, y, label=s)
        ax9.set_title("Latency CDF by Strategy"); ax9.legend()
    else:
        ax8.set_title("Latency Histogram: no read_logs")
        ax9.set_title("Latency CDF: no read_logs")
    savefig(fig, outdir, f"overview_{args.run_id}.png")

    # ============ 2) Latency 五分位综合 ============
    fig, ax = plt.subplots(figsize=(10,6))
    w=0.16
    ax.bar(xs-2*w, mdf["p50_ms"], w, label="P50")
    ax.bar(xs-1*w, mdf["p90_ms"], w, label="P90")
    ax.bar(xs+0*w, mdf["p95_ms"], w, label="P95")
    ax.bar(xs+1*w, mdf["p99_ms"], w, label="P99")
    ax.bar(xs+2*w, mdf["p999_ms"], w, label="P999")
    ax.set_title("Latency Percentiles"); ax.legend(); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    savefig(fig, outdir, f"latency_percentiles_{args.run_id}.png")

    # ============ 3) Total Cost（= release + read） ============
    fig, ax = plt.subplots(figsize=(10,6))
    bottom = np.zeros(len(labels))
    ax.bar(xs, mdf["on_release_cost"], bottom=bottom, label="On-release"); bottom += mdf["on_release_cost"].values
    ax.bar(xs, mdf["on_read_cost"], bottom=bottom, label="On-read")
    ax.plot(xs, mdf["total_cost"], marker="o", linestyle="--", label="Total")
    ax.set_title("Total Cost (= release + read)"); ax.legend(); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    savefig(fig, outdir, f"total_cost_{args.run_id}.png")

    # ============ 4) Precision / Recall / Fβ（仅选择性策略） ============
    has_pr = mdf["precision"].notna().any() or mdf["recall"].notna().any() or mdf["fbeta"].notna().any()
    if has_pr:
        sub = mdf[mdf["strategy_name"].isin(SEL_STRATEGIES)]
        if not sub.empty:
            lab = sub["strategy_name"].astype(str).tolist()
            x2 = np.arange(len(lab))
            fig, ax = plt.subplots(figsize=(10,6))
            w=0.25
            ax.bar(x2- w, sub["precision"].fillna(0), w, label="Precision")
            ax.bar(x2+ 0, sub["recall"].fillna(0),   w, label="Recall")
            ax.bar(x2+ w, sub["fbeta"].fillna(0),    w, label="Fβ")
            ax.set_title("Precision / Recall / Fβ (Selective Strategies)")
            ax.legend(); ax.set_xticks(x2); ax.set_xticklabels(lab, rotation=30, ha="right")
            savefig(fig, outdir, f"pr_fbeta_{args.run_id}.png")

    # ============ 5) 失败画像（堆叠+重试折线） ============
    fig, ax = plt.subplots(figsize=(10,6))
    dup = mdf.get("dup_key_err", pd.Series([0]*len(mdf))).values
    val = mdf.get("validate_err", pd.Series([0]*len(mdf))).values
    cas = mdf.get("cast_err", pd.Series([0]*len(mdf))).values
    oth = mdf.get("other_err", pd.Series([0]*len(mdf))).values
    ret = mdf.get("retry_count", pd.Series([0]*len(mdf))).values
    bottom = np.zeros(len(labels))
    ax.bar(xs, dup, bottom=bottom, label="dup"); bottom += dup
    ax.bar(xs, val, bottom=bottom, label="validate"); bottom += val
    ax.bar(xs, cas, bottom=bottom, label="cast"); bottom += cas
    ax.bar(xs, oth, bottom=bottom, label="other")
    ax.plot(xs, ret, marker="x", linestyle="--", label="retries")
    ax.set_title("Errors (stack) + Retries (line)"); ax.legend(); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    savefig(fig, outdir, f"errors_{args.run_id}.png")

    # ============ 6) 吞吐 ============
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(xs, mdf["throughput_docs_per_s"])
    ax.set_title("Throughput (docs/s)"); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    savefig(fig, outdir, f"throughput_{args.run_id}.png")

    # ============ 7) 每策略延迟直方图 + 箱线图 ============
    cli = MongoClient(MONGO_URI)
    for s in labels:
        sub = list(cli[DB_NAME][READLOGS].find({"run_id": args.run_id, "strategy_name": s}, {"_id":0, "lat_ms_total":1}))
        if not sub:
            # 空数据也生成占位图
            fig, ax = plt.subplots(figsize=(8,4)); ax.set_title(f"{s}: no read_logs"); savefig(fig, outdir, f"latency_hist_{s}_{args.run_id}.png")
            fig, ax = plt.subplots(figsize=(8,4)); ax.set_title(f"{s}: no read_logs"); savefig(fig, outdir, f"latency_box_{s}_{args.run_id}.png")
            continue
        vals = to_float_series(pd.Series([x.get("lat_ms_total", 0.0) for x in sub])).values
        # 直方
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(vals, bins=30)
        ax.set_title(f"Latency Histogram - {s}")
        savefig(fig, outdir, f"latency_hist_{s}_{args.run_id}.png")
        # 箱线（抽样）
        if len(vals) > args.boxplot_sample:
            idx = np.random.choice(len(vals), size=args.boxplot_sample, replace=False)
            vals = vals[idx]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.boxplot([vals], showfliers=False)
        ax.set_xticks([1]); ax.set_xticklabels([s])
        ax.set_title(f"Latency Boxplot - {s} (sample≤{args.boxplot_sample})")
        savefig(fig, outdir, f"latency_box_{s}_{args.run_id}.png")

    # ============ 8) 导出 CSV ============
    if args.csv_prefix:
        mdf.to_csv(f"{args.csv_prefix}_metrics.csv", index=False)
        if not bdf.empty:
            bdf.to_csv(f"{args.csv_prefix}_batches.csv", index=False)
        if not ldf.empty:
            ldf.to_csv(f"{args.csv_prefix}_readlogs.csv", index=False)
        print("Saved CSVs with prefix:", args.csv_prefix)

if __name__ == "__main__":
    main()
