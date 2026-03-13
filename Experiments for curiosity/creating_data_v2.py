import random
import pandas as pd
from pathlib import Path


N_SAMPLES = 100_000
X_MIN = -100.0
X_MAX = 100.0
Y_MIN = -100.0
Y_MAX = 100.0
NOISE = 4.0
OUTPUT_PATH = Path(__file__).parent / "raw_data_v2.csv"


def generate_data():
    x_range = X_MAX - X_MIN
    y_range = Y_MAX - Y_MIN
    x = [random.random() * x_range + X_MIN for _ in range(N_SAMPLES)]
    z = [random.random() * y_range + Y_MIN for _ in range(N_SAMPLES)]
    inputs = zip(x, z)
    outputs = [x ** 2 + random.random() * NOISE * random.choice([-1, 1]) + z ** 2 + random.random() * NOISE * random.choice([-1, 1])
               for (x, z) in inputs]
    return pd.DataFrame({"X": x, "Z": z, "Y": outputs})


if __name__ == "__main__":
    df = generate_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
