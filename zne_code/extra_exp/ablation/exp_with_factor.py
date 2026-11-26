# -*- coding: utf-8 -*-
"""把 zne.py 的因子修正也应用到「标准指数」上，用于回答审稿人。

zne.py 只把因子乘到 poly_exp 列；这里用同一套协议（Random(123) 选 3 条、
c=max(|exact/pred|,1)、取均值）对 exp 模型做一次，看看结论是否还成立。
上限 exp(0.001*n_gates) 在这里不会触发（因子最大 ~1.3，而 n_gates 上千），
脚本会核对这一点。
"""
import collections
import csv
import os
import random

from qiskit import QuantumCircuit

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    for fam in ("random", "ising_trotter"):
        rows = list(csv.DictReader(open(os.path.join(RES, fam, "per_circuit.csv"),
                                        encoding="utf-8")))
        by_entry = collections.defaultdict(dict)
        for r in rows:
            by_entry[int(r["entry"])][int(r["circuit"])] = r
        n_circ = max(len(v) for v in by_entry.values())
        sample = sorted(random.Random(123).sample(range(n_circ), min(3, n_circ)))

        tot, n = 0.0, 0
        facs = []
        root = os.path.normpath(os.path.join(HERE, "..", "..",
                                             "高斯-指数混合模型整合"))
        prefix = {"random": "random_circuits",
                  "ising_trotter": "ising_trotter_circuits"}[fam]
        for entry, d in sorted(by_entry.items()):
            cdir = os.path.join(root, prefix, "%s_%d" % (prefix, entry))
            ng = {}
            for i in sample:
                qc = QuantumCircuit.from_qasm_file(
                    os.path.join(cdir, "circuit_%d.qasm" % i))
                qc.remove_final_measurements()
                ng[i] = len(qc.data)
            cs = []
            for i in sample:
                r = d[i]
                ex, pr = float(r["exact"]), float(r["exp_pred"])
                c = abs(ex / pr) if (pr != 0.0 and ex != 0) else 1.0
                c = max(c, 1.0)
                c = min(c, __import__("math").exp(ng[i] * 0.001))
                cs.append(c)
            c = sum(cs) / len(cs)
            facs.append(c)
            for r in d.values():
                pred = float(r["exp_pred"]) * c
                tot += abs(pred - float(r["exact"]))
                n += 1
        print("%-14s exp with factor = %.5f   (factor: min %.4f median %.4f max %.4f)"
              % (fam, tot / n, min(facs), sorted(facs)[len(facs) // 2], max(facs)))


if __name__ == "__main__":
    main()
