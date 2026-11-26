
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeLima, FakeQuito, FakeJakarta, FakeNairobi

def create_fixed_noise_model():
    backends = {
        "FakeLima": FakeLima(),
        "FakeQuito": FakeQuito(),
        "FakeJakarta": FakeJakarta(),
        "FakeNairobi": FakeNairobi()
    }

    # select backed
    selected_backend = backends["FakeQuito"]
    noise_model = NoiseModel.from_backend(selected_backend)
    return noise_model
