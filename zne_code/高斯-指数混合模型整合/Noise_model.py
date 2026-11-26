"""Device noise model used by the density-matrix ZNE path.

The paper's "quantum-computer-like" noise model is the calibration snapshot of
the IBM ``FakeQuito`` device, read through ``NoiseModel.from_backend``.

WARNING for future edits
------------------------
This module intentionally uses the **Qiskit < 1.0** API surface: the V1 fake
backend ``qiskit.providers.fake_provider.FakeQuito`` (which exposes
``.configuration()`` / ``.properties()``).  It must not be switched to
``FakeQuitoV2`` / ``qiskit_ibm_runtime.fake_provider`` -- that changes the
reported calibration data and therefore every simulated number.
"""

from __future__ import annotations

from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import (
    FakeJakarta,
    FakeLima,
    FakeNairobi,
    FakeQuito,
)


def create_fixed_noise_model() -> NoiseModel:
    """Return the fixed ``NoiseModel`` for the paper's simulations.

    ``FakeQuito`` is the selected device.  The other three backends are kept in
    the mapping because the original code built them as well; only the selected
    one is passed to ``NoiseModel.from_backend``.  Constructing a fake backend is
    side-effect free and draws no random numbers.
    """
    backends = {
        "FakeLima": FakeLima(),
        "FakeQuito": FakeQuito(),
        "FakeJakarta": FakeJakarta(),
        "FakeNairobi": FakeNairobi(),
    }

    # select backend
    selected_backend = backends["FakeQuito"]
    noise_model = NoiseModel.from_backend(selected_backend)
    return noise_model
