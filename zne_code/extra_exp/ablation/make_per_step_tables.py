# -*- coding: utf-8 -*-
"""把消融实验按 step 拆开保存，并标出每个 step 上哪个模型最好。

输入：results/{random,ising_trotter,grover}/summary.csv（已存在的逐 step + 汇总结果）
输出（写到 results/per_step/）：
  - per_step_long.csv        family, step, variant, mean/median/max, n_rejected
  - random_wide.csv          step 为行，各变体的 mean_abs_err 为列
  - ising_trotter_wide.csv
  - grover_wide.csv
  - winners.csv              family, step, best, best_value, second, second_value
只用已有结果，不重跑任何拟合。
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "per_step")
FAMS = ("random", "ising_trotter", "grover")
# 参与“谁最好”比较的候选（与论文 Table 6 的五个消融变体一致）
CAND = ("A", "B", "C", "D", "E")


def read(fam):
    rows = []
    with open(os.path.join(RES, fam, "summary.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["entry"] in ("ALL", ""):
                continue                      # 汇总行不要
            rows.append(r)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    long_rows, winners = [], []

    for fam in FAMS:
        rows = read(fam)
        steps = sorted({int(r["step"]) for r in rows})
        variants = sorted({r["variant"] for r in rows})
        wide = {}
        for r in rows:
            step = int(r["step"])
            wide.setdefault(step, {})[r["variant"]] = r

        # 长表
        for step in steps:
            for v in variants:
                r = wide[step].get(v)
                if not r:
                    continue
                long_rows.append(dict(
                    family=fam, step=step, variant=v,
                    n_circuits=r["n_circuits"],
                    mean_abs_err=float(r["mean_abs_err"]),
                    median_abs_err=float(r["median_abs_err"]),
                    max_abs_err=float(r["max_abs_err"]),
                    n_rejected=r["n_rejected_gt_1.2"]))

        # 宽表
        cols = ["step"] + [v for v in ("NOISY", "EXP", "A", "B", "D", "E", "C")
                           if v in variants]
        with open(os.path.join(OUT, fam + "_wide.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for step in steps:
                row = [step]
                for v in cols[1:]:
                    r = wide[step].get(v)
                    row.append("" if not r else "%.6g" % float(r["mean_abs_err"]))
                w.writerow(row)

        # 最优
        for step in steps:
            cand = [(v, float(wide[step][v]["mean_abs_err"])) for v in CAND
                    if v in wide[step]]
            if not cand:
                continue
            cand.sort(key=lambda t: t[1])
            best_v, best_val = cand[0]
            second_v, second_val = cand[1] if len(cand) > 1 else ("", float("nan"))
            winners.append(dict(family=fam, step=step, best=best_v,
                                best_value=best_val, second=second_v,
                                second_value=second_val))

    with open(os.path.join(OUT, "per_step_long.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)

    with open(os.path.join(OUT, "winners.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(winners[0].keys()))
        w.writeheader()
        w.writerows(winners)

    # 控制台报告
    from collections import Counter
    for fam in FAMS:
        ws = [x for x in winners if x["family"] == fam]
        cnt = Counter(x["best"] for x in ws)
        print("=" * 64)
        print("%s  (%d steps)  胜出次数: %s" % (fam, len(ws), dict(cnt)))
        print("%4s %-6s %10s %-6s %10s" % ("step", "best", "value", "2nd", "value"))
        for x in ws:
            print("%4d %-6s %10.5f %-6s %10.5f"
                  % (x["step"], x["best"], x["best_value"],
                     x["second"], x["second_value"]))
    print("\nsaved to", OUT)


if __name__ == "__main__":
    main()
