import numpy as np

def create_restartable_random_system(base_seed=42):
    """Create a restartable random number system."""

    class RestartableRandom:
        def __init__(self, seed=base_seed):
            self.base_seed = seed
            self.rng = np.random.RandomState(seed)
            self.call_count = 0

        def rand(self, *args):
            self.call_count += 1
            return self.rng.rand(*args)

        def randn(self, *args):
            self.call_count += 1
            return self.rng.randn(*args)

        def randint(self, *args):
            self.call_count += 1
            return self.rng.randint(*args)

        def choice(self, *args):
            self.call_count += 1
            return self.rng.choice(*args)

        def restart(self, new_seed=None):
            """Restart the random sequence."""
            if new_seed is None:
                new_seed = self.base_seed
            self.rng = np.random.RandomState(new_seed)
            self.call_count = 0
            print(f"Random number generator restarted with seed: {new_seed}")

        def get_state(self):
            return self.rng.get_state()

        def set_state(self, state):
            self.rng.set_state(state)

        def uniform(self, low=0.0, high=1.0, size=None):
            """Sample uniformly from [low, high)."""
            self.call_count += 1
            return self.rng.uniform(low, high, size)

    return RestartableRandom(base_seed)

# Usage example
def test_restartable_system():
    print("\n=== Testing Restartable Random System ===")
    rng = create_restartable_random_system(123)

    print("First run:")
    for i in range(3):
        print(f"  Random number {i}: {rng.rand()}")

    print(f"Total call count: {rng.call_count}")

    # Restart - reset the counter
    rng.restart()

    print("After restart:")
    for i in range(3):
        print(f"  Random number {i}: {rng.rand()}")

    print(f"Call count after restart: {rng.call_count}")

    test_restartable_system()