# Environment setup

Verified working environment, created and tested on this machine.

```
conda env name : zne_qiskit
python         : 3.11.17
location       : C:\Users\wangt\anaconda3\envs\zne_qiskit
```

## Create it

```bash
conda create -n zne_qiskit python=3.11 -y
conda activate zne_qiskit
pip install -r requirements.txt
conda install -n zne_qiskit -c conda-forge graphviz -y   # binaries for plot_error_map
```

Or, equivalently, from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate zne_qiskit
```

`requirements.txt` and `environment.yml` carry the same pins.

## Activate and run

```bash
conda activate zne_qiskit
cd E:/论文修改/zne_code
python main.py --list      # what can run, and in which mode
python main.py --all       # the cached-data pipeline (~21 s)
```

You **must** activate the environment (not just call its `python.exe`
directly): `plot_error_map` shells out to the Graphviz `dot` executable, which
conda places in `envs/zne_qiskit/Library/bin` and only puts on `PATH` on
activation.  `main.py` also exports `MPLBACKEND=Agg` and
`PYTHONIOENCODING=utf-8` for every step it launches, so it does not matter which
backend or console encoding your shell has.

---

## The pins, and why each one

The project was written against the **pre-Qiskit-1.0 API**.  That is not a guess:
four independent calls in the tree only exist before 1.0.

| Code location | Call | Removed / renamed in |
|---|---|---|
| `zne.py:10`, `tools.py:3`, `stable.py:3`, `grover.py:11`, `noise_exp_compute.py:6`, `example_random/clifford_road_sample.py:18` | `from qiskit import Aer` | Qiskit 1.0 |
| `Noise_model.py:3`, `zne.py:13`, `random_generator.py:6`, `trotter_generate.py:6`, `Device_properties.py:12`, `stable.py:20`, `zne_grover_shots.py:20` | `from qiskit.providers.fake_provider import FakeLima, FakeQuito, ...` (V1 fake backends) | V1 backends removed in Qiskit 2.0; renamed/moved afterwards |
| `Device_properties.py:25-26` | `backend.configuration()` / `backend.properties()` | V1-only API |
| `pk_file_deployment.py:4` | `from qiskit.circuit import qpy_serialization` | renamed to `qiskit.qpy` in Qiskit 1.0 |

`qiskit 0.46.3` (== `qiskit-terra 0.46.3`) is the **last** release of the
pre-1.0 line, and the newest release where all four of the above exist at once.
It was confirmed by inspecting the wheels directly:

```
qiskit_terra-0.46.1-cp38-abi3-win_amd64.whl
  qiskit/__init__.py                     -> class AerWrapper  ("qiskit.Aer")
  qiskit/providers/fake_provider/__init__.py -> FakeLima, FakeQuito, FakeJakarta, FakeNairobi
  qiskit/circuit/qpy_serialization.py    -> present
```

For comparison, `qiskit 1.0.2` already lacks `Aer`, `FakeLima` and
`qpy_serialization`, and `qiskit 2.x` lacks `Aer` and the V1 `FakeLima` entirely.

### Package-by-package

| Pin | Why |
|---|---|
| `python=3.11` | Required by `qiskit-aer 0.14.2` Windows wheels **and** supported by `qiskit-terra 0.46`.  NOT 3.14 (the user's current interpreter, which has no compatible Qiskit). |
| `qiskit==0.46.3` | Metapackage; the newest pre-1.0 release. |
| `qiskit-terra==0.46.3` | Pinned explicitly because `qiskit==0.46.3` only re-exports it, and a stray newer `qiskit-terra` breaks the install (this is exactly the failure seen in the local `machine_learning` env, which has `qiskit 2.2.3` and `qiskit-terra 0.24.1` side by side and raises "Qiskit is installed in an invalid environment that has both Qiskit >=1.0 and an earlier version"). |
| `qiskit-aer==0.14.2` | Last Aer release compatible with `qiskit-terra 0.46`; supplies the `aer_simulator_density_matrix`, `aer_simulator`, `qasm_simulator` and `statevector_simulator` backends the code requests. Its `Requires-Dist: qiskit >=0.45.2` is satisfied. |
| `numpy==1.26.4` | `qiskit-aer 0.14.2` was built against the NumPy 1.x ABI (released before NumPy 2.0), so NumPy 2 will not import it. 1.26.4 is the last 1.x. |
| `scipy==1.13.1` | Provides `curve_fit` / `stats.lognorm`; compatible with NumPy 1.26. |
| `matplotlib==3.9.4` | All figures go through `matplotlib`; 3.9.4 is the version the reference figures were reproduced with (pixel-identical, see README.md). |
| `pandas==2.2.3` | Used only by `Device_properties.py` for the CSV tables. |
| `networkx==3.2.1` | Used only by `Device_properties.py` for the coupling-map drawing. |
| `tqdm==4.67.1` | Imported by `random_generator.py` and `trotter_generate.py`. |
| `torch==2.7.0` | Imported at module level by `random_generator.py` (it calls `torch.manual_seed` in `fix_random_seed`). Only needed if you run the random-circuit generation step; every analysis step works without it. |
| `pylatexenc==2.10` | Required by Qiskit's `MatplotlibDrawer` (`qc.draw('mpl')` in `run_circuit.py`). |
| `graphviz==0.21` + Graphviz binaries | Required by `plot_error_map` in `Device_properties.py`. The pip package alone is not enough; the `dot` binary must be on `PATH`, hence the conda install. |
| `seaborn==0.13.2` | Also required by `plot_error_map`. |

### Do NOT install `qiskit-ibm-runtime`

The archive's code **never imports it**.  Every published `qiskit-ibm-runtime`
requires `qiskit>=1.0`, so installing it would silently upgrade Qiskit and break
`from qiskit import Aer` -- reintroducing exactly the breakage this environment
fixes.

(Only a *sibling* file outside this archive,
`Hybrid_Gaussian_exponential_zero_noise_extrapolation_for_periodic_circuits__4_ (1)/run_grover_stability.py`,
uses `from qiskit_ibm_runtime.fake_provider import FakeLimaV2`.  That file is a
modern rewrite living with the paper, not with this code, and needs a separate
Qiskit >= 1.x environment.  The code in this archive is authoritative for this
archive.)

---

## Smoke test (this is the actual output on this machine)

```bash
conda activate zne_qiskit
python -c "
import qiskit, qiskit_aer, numpy, scipy, matplotlib, pandas, networkx
from qiskit import Aer
from qiskit.providers.fake_provider import FakeLima, FakeQuito, FakeJakarta, FakeNairobi
from qiskit_aer.noise import NoiseModel
from qiskit.circuit import qpy_serialization
print(qiskit.__version__, qiskit_aer.__version__, numpy.__version__)
b = FakeLima()
print(type(b).__name__, b.configuration().backend_name, b.configuration().n_qubits)
print(NoiseModel.from_backend(b))
print(Aer.get_backend('aer_simulator_density_matrix').name)
"
```

```
qiskit        0.46.3
qiskit_aer    0.14.2
numpy         1.26.4
scipy         1.13.1
matplotlib    3.9.4
pandas        2.2.3
networkx      3.2.1
FakeLima type FakeLima
backend_name  fake_lima
n_qubits      5
noise model   NoiseModel:
  Basis gates: ['cx', 'id', 'reset', 'rz', 'sx', 'x']
  Instructions with noise: ['reset', 'cx', 'measure', 'x', 'sx', 'id']
  Qubits with noise: [0, 1, 2, 3, 4]
  ...
Aer backend   aer_simulator_density_matrix
```

Note that `FakeLima()` resolves to the **V1** backend (`.configuration()`,
`.properties()` available) -- exactly what the code expects.
