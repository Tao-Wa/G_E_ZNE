"""
Vectorised re-implementation of the Pauli-path sampler.

This module reproduces, in a batched NumPy form, the *exact* algorithm of
``example_random/clifford_road_sample.sample_path_with_observable`` from the
paper's code tree.  It is a speed-only rewrite used by ``run_sweep.py``; the
original function remains untouched and importable, and can be selected with
``--sampler original``.

Fidelity strategy (why this is not "re-deriving the physics")
-------------------------------------------------------------
Every lookup table used here is produced *by calling the original functions*:

* the Clifford propagation tables come from
  ``clifford_road_sample.propagate_pauli_through_gate``,
* the rotation branch tables (new Pauli letter, replacement gate, accumulated
  amplitude, accumulated branching weight ``p2``) come from
  ``clifford_road_sample.sample_rz / sample_rx / sample_ry`` called with a stub
  RNG pinned to ``u = 0.0`` (branch 0) and ``u = 0.999999`` (branch 1),
* the QASM line formatting is copied verbatim from
  ``clifford_road_sample.sample_path_with_observable``.

The batched loop then applies exactly those tables.  Nothing about the gate
algebra, the branch probabilities or the gate replacement rules is transcribed
by hand.

What differs from the sequential sampler (documented deviations)
----------------------------------------------------------------
1. **RNG stream layout.**  ``Attempts`` are processed in batches of
   ``batch_attempts``; at every non-Clifford instruction one
   ``rng.uniform(0, 1, size=B)`` vector is drawn instead of one scalar per
   attempt.  Every individual draw is still an i.i.d. ``U(0,1)`` used with the
   same thresholds (the sequential code's ``idx = 0 if u < 0.5 else 1``, or
   ``idx = 0 if u < probs[0] else 1`` in the near-deterministic case), so the
   per-path distribution is unchanged; only *which* uniform goes to which
   attempt changes.  A uniform draw is also taken for attempts whose Pauli
   letter does not branch (the sequential code consumes none there), which
   shifts the stream but not any individual draw's distribution.
2. **No intra-call ``seen_paths`` de-duplication.**  The sequential sampler
   suppresses a path that has already been emitted inside the same call, so a
   call returns distinct paths with (in the many-call limit) equal multiplicity;
   the batched sampler returns i.i.d. draws, whose multiplicity is proportional
   to the sampling probability ``qtilde``.  With importance weights
   ``w = 1/qtilde`` the two give the same self-normalised weighted empirical
   distribution, which is the only object the estimator uses.
3. **No ``max_attempts`` cap.**  Rejected attempts are simply discarded and the
   batch continues; the sequential cap can drop a tiny number of samples.
4. Attempts are not pruned early (the original only tests validity after the
   full backward pass), so the accepted set is drawn from the same conditional
   distribution.

Validation
----------
``run_sweep.py --validate-sampler`` compares the two samplers.  For ``m = 3``
(504 reachable paths) and ``m = 4`` (12 664 reachable paths) the batched sampler
reproduces the *identical* reachable path set: same number of distinct paths and
the same multiset of ``(qtilde, qasm-line-sequence)`` pairs as the saturating run
of the original sampler.
"""

from __future__ import annotations

import numpy as np

import clifford_road_sample as crs

SYM = "IXYZ"
CODE = {s: i for i, s in enumerate(SYM)}

SINGLE_QUBIT_CLIFFORDS = ("x", "h", "s", "sdg", "sx")
ROTATIONS = ("rz", "rx", "ry")


class _FixedRng:
    """Stub RNG pinning every ``uniform`` draw to one value (branch selection)."""

    __slots__ = ("u",)

    def __init__(self, u):
        self.u = float(u)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return self.u
        return np.full(size, self.u)


def _pauli_string(nq, letters):
    """Pauli label of length nq with ``letters`` (list of codes) at the front."""
    out = ["I"] * nq
    for i, c in enumerate(letters):
        out[i] = SYM[c]
    return "".join(out)


def fmt_line(name, params, qubits):
    """Verbatim copy of the QASM line formatting in sample_path_with_observable."""
    if len(params) > 0:
        p = ",".join(str(x) for x in params)
        return f"{name}({p}) q[{qubits[0]}];"
    if len(qubits) == 1:
        return f"{name} q[{qubits[0]}];"
    return f"{name} q[{qubits[0]}], q[{qubits[1]}];"


class VectorSampler:
    """Batched Pauli-path sampler for one circuit (see module docstring)."""

    def __init__(self, qc, observable):
        self.nq = int(qc.num_qubits)
        self.obs = observable.to_label() if hasattr(observable, "to_label") else str(observable)
        self.instructions = list(qc.data)          # (instruction, qargs, cargs)
        self.L = len(self.instructions)
        if len(self.obs) != self.nq:
            raise ValueError("observable length must equal the number of qubits")

        self._op_keys = []      # opcode -> (name, params_tuple, qubits_tuple)
        self._op_ids = {}       # (name, params_tuple, qubits_tuple) -> opcode
        self._lines = []        # opcode -> QASM line
        self.per_inst = []      # per instruction descriptor

        self._build()

    # ------------------------------------------------------------------ #
    def _register(self, name, params, qubits):
        key = (str(name), tuple(str(p) for p in params), tuple(int(q) for q in qubits))
        code = self._op_ids.get(key)
        if code is None:
            code = len(self._op_keys)
            self._op_ids[key] = code
            self._op_keys.append(key)
            self._lines.append(fmt_line(key[0], list(params), list(key[2])))
        return code

    def opcode_line(self, code):
        return self._lines[code]

    # ------------------------------------------------------------------ #
    def _build(self):
        for inst, qargs, _cargs in self.instructions:
            name = inst.name.lower()
            qubits = [q.index for q in qargs]
            if name == "measure":
                self.per_inst.append({"kind": "measure"})
                continue
            if name == "barrier":
                code = self._register("barrier", [], qubits)
                self.per_inst.append({"kind": "const", "opcode": code})
                continue
            if name in ROTATIONS:
                self.per_inst.append(self._rotation_entry(name, float(inst.params[0]),
                                                          qubits))
                continue
            if name == "cx":
                if len(qubits) != 2:
                    raise NotImplementedError("cx must act on two qubits")
                self.per_inst.append(self._cx_entry(qubits))
                continue
            if name in SINGLE_QUBIT_CLIFFORDS:
                if len(qubits) != 1:
                    raise NotImplementedError(f"{name} must act on one qubit")
                self.per_inst.append(self._clifford1_entry(name, qubits[0]))
                continue
            raise NotImplementedError(f"gate {name!r} not supported by the vector sampler")

    # ------------------------------------------------------------------ #
    def _rotation_entry(self, name, theta, qubits):
        """Branch table for one rz/rx/ry, built by calling the original function."""
        fn = {"rz": crs.sample_rz, "rx": crs.sample_rx, "ry": crs.sample_ry}[name]
        q = qubits[0]

        new_letter = np.zeros((4, 2), dtype=np.int8)
        opcode = np.zeros((4, 2), dtype=np.int32)
        wfac = np.ones((4, 2), dtype=np.float64)
        pfac = np.ones((4, 2), dtype=np.float64)
        for col in range(4):
            # only qubit q matters for a single-qubit gate, the rest stay 'I'
            base = "I" * q + SYM[col] + "I" * (self.nq - q - 1)
            for idx, u in enumerate((0.0, 0.999999)):
                newP, chosen_gate, prob, p2 = fn(theta, base, [q], _FixedRng(u))
                new_letter[col, idx] = CODE[newP[q]]
                opcode[col, idx] = self._register(chosen_gate[0], chosen_gate[1],
                                                  chosen_gate[2])
                wfac[col, idx] = p2
                pfac[col, idx] = prob

        # non-branching case: use a letter that never triggers (I on qubit q)
        base = "I" * self.nq
        _newP, chosen_gate, _prob, _p2 = fn(theta, base, [q], _FixedRng(0.0))
        idle_code = self._register(chosen_gate[0], chosen_gate[1], chosen_gate[2])

        # which letters trigger a branch (direction-independent amplitudes)
        amps = np.abs([np.cos(theta), np.sin(theta)])
        probs = amps / amps.sum()
        return {
            "kind": "rot", "q": q, "name": name, "theta": theta,
            "new_letter": new_letter, "opcode": opcode, "wfac": wfac, "pfac": pfac,
            "idle_code": idle_code,
            "triggers": (new_letter[:, 0] != np.arange(4)) | (new_letter[:, 1] != np.arange(4)),
            "degenerate": bool(probs[0] < 1e-4 or probs[1] < 1e-4),
            "thr_degenerate": float(probs[0]),
        }

    # ------------------------------------------------------------------ #
    def _clifford1_entry(self, name, q):
        new_letter = np.zeros(4, dtype=np.int8)
        sign = np.ones(4, dtype=np.int8)
        for col in range(4):
            base = "I" * q + SYM[col] + "I" * (self.nq - q - 1)
            nP, s = crs.propagate_pauli_through_gate(base, name, [q])
            new_letter[col] = CODE[nP[q]]
            sign[col] = s
        code = self._register(name, [], [q])
        return {"kind": "clifford1", "q": q, "new_letter": new_letter,
                "sign": sign, "opcode": code}

    # ------------------------------------------------------------------ #
    def _cx_entry(self, qubits):
        q0, q1 = qubits
        c_new = np.zeros((4, 4), dtype=np.int8)
        t_new = np.zeros((4, 4), dtype=np.int8)
        sign = np.ones((4, 4), dtype=np.int8)
        for cc in range(4):
            for ct in range(4):
                base = ["I"] * self.nq
                base[q0] = SYM[cc]
                base[q1] = SYM[ct]
                nP, s = crs.propagate_pauli_through_gate("".join(base), "cx", [q0, q1])
                c_new[cc, ct] = CODE[nP[q0]]
                t_new[cc, ct] = CODE[nP[q1]]
                sign[cc, ct] = s
        code = self._register("cx", [], [q0, q1])
        return {"kind": "clifford2", "q0": q0, "q1": q1, "c_new": c_new,
                "t_new": t_new, "sign": sign, "opcode": code}

    # ------------------------------------------------------------------ #
    def sample(self, n_accept, rng, batch_attempts=4096, max_rounds=None,
               progress=None, max_attempts=None):
        """Draw until ``n_accept`` valid paths are collected.

        ``max_attempts`` is a safety cap on the total number of attempts
        (default ``50 * n_accept / acceptance_floor``-free: ``None`` means
        ``500 * n_accept``).  The original sampler has the same kind of guard
        via ``max_attempts=300`` per requested sample; without it a circuit
        that admits no valid path would loop forever.

        Returns a dict of arrays:
            qtilde   : float64, product of the branching probabilities (the
                       original's ``weight`` field, i.e. ``pauli_weight``)
            prob     : float64, product of the branch amplitudes*sign
                       (the original's ``probability`` field, i.e. ``weigh``)
            opcode   : int32 (n_accept, L), the gate chosen at every instruction
                       in *backward* order (the QASM lines are produced by
                       reversing this, exactly as in the original)
            final_pauli : the Pauli string after the full backward pass
            attempts : total attempts spent
        """
        nq, L = self.nq, self.L
        instr = self.per_inst

        out_qtilde, out_prob, out_opc, out_pauli = [], [], [], []
        got = 0
        attempts = 0
        rounds = 0

        if max_attempts is None:
            max_attempts = 500 * max(int(n_accept), 1)
        while got < n_accept:
            rounds += 1
            if max_rounds is not None and rounds > max_rounds:
                break
            if attempts >= max_attempts:
                print(f"WARNING: fast_sampler hit max_attempts={max_attempts} after "
                      f"{got}/{n_accept} accepted paths (acceptance "
                      f"{got / max(attempts, 1):.3g}); returning what was found.")
                break
            B = int(batch_attempts)
            attempts += B
            P = np.empty((B, nq), dtype=np.int8)
            P[:] = np.array([CODE[c] for c in self.obs], dtype=np.int8)
            w = np.ones(B, dtype=np.float64)
            prob = np.ones(B, dtype=np.float64)
            opcode_hist = np.zeros((B, L), dtype=np.int32)

            for i in range(L - 1, -1, -1):
                d = instr[i]
                kind = d["kind"]
                if kind == "measure":
                    continue
                if kind == "const":
                    opcode_hist[:, i] = d["opcode"]
                    continue
                if kind == "clifford1":
                    q = d["q"]
                    col = P[:, q]
                    prob = prob * d["sign"][col]
                    P[:, q] = d["new_letter"][col]
                    opcode_hist[:, i] = d["opcode"]
                    continue
                if kind == "clifford2":
                    q0, q1 = d["q0"], d["q1"]
                    cc = P[:, q0].astype(np.intp)
                    ct = P[:, q1].astype(np.intp)
                    prob = prob * d["sign"][cc, ct]
                    P[:, q0] = d["c_new"][cc, ct]
                    P[:, q1] = d["t_new"][cc, ct]
                    opcode_hist[:, i] = d["opcode"]
                    continue
                if kind == "rot":
                    q = d["q"]
                    col = P[:, q].astype(np.intp)
                    active = d["triggers"][col]
                    u = rng.uniform(0.0, 1.0, size=B) if rng is not None else \
                        np.zeros(B)
                    thr = d["thr_degenerate"] if d["degenerate"] else 0.5
                    idx = (u >= thr).astype(np.intp)
                    nl = d["new_letter"][col, idx]
                    oc = d["opcode"][col, idx]
                    wf = d["wfac"][col, idx]
                    pf = d["pfac"][col, idx]
                    P[:, q] = np.where(active, nl, col).astype(np.int8)
                    opcode_hist[:, i] = np.where(active, oc, d["idle_code"])
                    w = w * np.where(active, wf, 1.0)
                    prob = prob * np.where(active, pf, 1.0)
                    continue
                raise AssertionError(kind)

            valid = ~np.any((P == CODE["X"]) | (P == CODE["Y"]), axis=1)
            take = np.flatnonzero(valid)
            had = len(out_qtilde)
            n_take = len(take)
            if n_take:
                out_qtilde.append(w[take])
                out_prob.append(prob[take])
                out_opc.append(opcode_hist[take])
                out_pauli.append(P[take])
                got += n_take
            if progress is not None:
                progress(rounds, got, n_accept, had)

        total = sum(a.shape[0] for a in out_qtilde)
        if total == 0:
            raise RuntimeError("no valid path sampled")
        # the last round overshoots: trim to exactly n_accept
        q = np.concatenate(out_qtilde)[:n_accept]
        p = np.concatenate(out_prob)[:n_accept]
        o = np.concatenate(out_opc, axis=0)[:n_accept]
        f = np.concatenate(out_pauli, axis=0)[:n_accept]
        return {
            "qtilde": q, "prob": p, "opcode": o, "final_pauli": f,
            "attempts": attempts, "n_accepted_total": int(total),
        }

    # ------------------------------------------------------------------ #
    def qasm_lines(self, opcode_row):
        """QASM lines of one accepted path, in the original's forward order.

        ``opcode_row[i]`` is the gate chosen for instruction ``i`` of the
        circuit, so reading the row left to right already gives the forward
        order the original produces after its ``list(reversed(clifford_ops))``.
        No reversal is applied here.
        """
        lines = self._lines
        return [lines[c] for c in opcode_row]

    def qasm_string(self, opcode_row):
        return "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[%d];\n" % self.nq \
            + "".join(ln + "\n" for ln in self.qasm_lines(opcode_row))


def hash_opcode_row(row):
    """Stable 64-bit fingerprint of an opcode row (used to identify paths)."""
    return hash(row.tobytes())
