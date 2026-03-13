import random
import pandas as pd
from pathlib import Path

N_SAMPLES = 100_000
X_MIN = -100.0
X_MAX = 100.0
NOISE = 4.0
OUTPUT_PATH = Path(__file__).parent / "raw_data.csv"


def generate_data():
    x_range = X_MAX - X_MIN
    inputs = [random.random() * x_range + X_MIN for _ in range(N_SAMPLES)]
    outputs = [x ** 2 + random.random() * NOISE * random.choice([-1, 1])
               for x in inputs]
    return pd.DataFrame({"X": inputs, "Y": outputs})


if __name__ == "__main__":
    df = generate_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
