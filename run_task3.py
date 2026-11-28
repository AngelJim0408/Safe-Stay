
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import llm_compare as lc  # noqa: E402

PROVIDERS = ["groq", "gemini"]          # main LLM + second model from a different provider
MODES = ["no_data", "with_data"]


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def main(refresh=False, offline=False):
    config.ensure_dirs()
    real = lc.real_monthly_averages()
    names = sorted(real["neighbourhood"])
    data_csv = (real[["neighbourhood", "real_monthly_avg"]]
                .assign(real_monthly_avg=lambda d: d["real_monthly_avg"].round(1))
                .sort_values("neighbourhood")
                .rename(columns={"real_monthly_avg": "avg_incidents_per_month"})
                .to_csv(index=False))
    prompts = {"no_data": lc.build_prompt(names),
               "with_data": lc.build_prompt(names, data_csv)}

    answers, comparison, metrics_all = [], [], []
    for provider in PROVIDERS:
        for mode in MODES:
            print(f"{provider} / {mode} ...", flush=True)
            rec = lc.ask(provider, mode, prompts[mode], names, refresh=refresh, offline=offline)
            answers.append(rec)
            if not rec.get("valid"):
                print(f"  no valid answer: {rec['attempts'][-1]['problems'][:3]}")
                continue
            matched, invented = lc.match_names(rec["parsed"], names)
            rows, m = lc.compare(real, matched, invented)
            for r in rows:
                comparison.append({"model": f"{provider}:{rec['model']}", "mode": mode, **r})
            metrics_all.append({"provider": provider, "model": rec["model"], "mode": mode,
                                "retrieved_at": rec["retrieved_at"],
                                "from_cache": rec.get("from_cache", False), **m})
            print(f"  spearman={fmt(m['spearman_rating_vs_real'])}  top5={m['top5_overlap']}/5  "
                  f"median_err={fmt(m['median_abs_pct_error'], 1)}%  "
                  f"skipped={len(m['skipped'])}  invented={len(m['invented'])}")

    out = config.OUTPUT_DIR
    (out / "t3_llm_answers.json").write_text(
        json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(comparison).to_csv(out / "t3_comparison.csv", index=False)
    (out / "t3_metrics.txt").write_text(render_metrics(real, metrics_all), encoding="utf-8")
    print("wrote", out / "t3_llm_answers.json", out / "t3_comparison.csv", out / "t3_metrics.txt")


def render_metrics(real, metrics_all):
    lines = ["Task 3: LLM crime levels vs. real data (San Francisco, 2022-2024)", "=" * 64,
             f"Real data: {len(real)} neighbourhoods, average incidents per month over 36 months "
             "(Task 1 table, missing months counted as 0).",
             "Real top 5: " + ", ".join(real.nsmallest(5, "real_rank")["neighbourhood"]), "",
             f"{'model':<34}{'mode':<11}{'spearman':>9}{'top5':>6}{'med.err%':>10}"
             f"{'skipped':>9}{'invented':>10}",
             "-" * 89]
    for m in metrics_all:
        lines.append(f"{m['provider'] + ':' + m['model']:<34}{m['mode']:<11}"
                     f"{fmt(m['spearman_rating_vs_real']):>9}{m['top5_overlap']:>4}/5"
                     f"{fmt(m['median_abs_pct_error'], 1):>10}"
                     f"{len(m['skipped']):>9}{len(m['invented']):>10}")
    lines += ["", "Columns: spearman = Spearman rank correlation between the LLM's 1-5 rating "
              "and the real monthly average; top5 = real top-5 neighbourhoods also in the LLM's "
              "top 5 (rating, ties broken by its estimate); med.err% = median absolute "
              "percentage error of the incidents-per-month estimates.", ""]
    for m in metrics_all:
        lines += [f"[{m['provider']}:{m['model']} / {m['mode']}]  retrieved {m['retrieved_at']} UTC"
                  + (" (cached)" if m["from_cache"] else ""),
                  f"  Spearman (estimate vs real): {fmt(m['spearman_estimate_vs_real'])}",
                  f"  LLM top 5: {', '.join(m['llm_top5'])}",
                  f"  Skipped: {', '.join(m['skipped']) or 'none'}",
                  f"  Invented: {', '.join(m['invented']) or 'none'}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and ask again")
    ap.add_argument("--offline", action="store_true", help="use cached answers only")
    a = ap.parse_args()
    main(refresh=a.refresh, offline=a.offline)
