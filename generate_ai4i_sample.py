"""Generate an evaluation sample that follows the documented AI4I 2020 generation process.

The UCI AI4I 2020 Predictive Maintenance Dataset is itself synthetic. Its documentation describes how each column
was produced, and this script re-implements that description so the pipeline notebook can run when the original
file cannot be downloaded. The output has the same columns, value ranges and failure rules, but it is NOT the
original file: results computed on it are for evaluating the pipeline, not findings about the published data.

Documented process (UCI dataset page, Matzka 2020):
* type: L / M / H in proportions 50% / 30% / 20%; product ID = type letter + serial number
* air temperature: random walk, normalised to a standard deviation of 2 K around 300 K
* process temperature: random walk added to the air temperature plus 10 K (calibrated here to the published
  sd of 1.48 K, its strong correlation with air temperature, and the published heat-failure frequency)
* rotational speed: derived from a power level with normally distributed noise (calibrated here as a hyperbolic
  speed-torque curve matching the published mean 1539 rpm, sd 179 rpm, right skew and -0.88 correlation)
* torque: normal around 40 Nm with sd 10 Nm, within the published range 3.8-76.6 Nm
* tool wear: each process adds 5 / 3 / 2 minutes for H / M / L; tools are replaced or fail at a random wear
  between 200 and 240 minutes
* failure modes: TWF (tool fails at its replacement point), HDF (temperature gap < 8.6 K and speed < 1380 rpm),
  PWF (power < 3500 W or > 9000 W), OSF (wear x torque > 11,000 / 12,000 / 13,000 min*Nm for L / M / H),
  RNF (0.1% random). Machine failure = 1 if any mode occurs.

Usage:  python generate_ai4i_sample.py [output.csv] [--rows N] [--seed S]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OSF_LIMIT = {"L": 11_000, "M": 12_000, "H": 13_000}
WEAR_PER_PROCESS = {"L": 2, "M": 3, "H": 5}
TOOL_FAIL_SHARE = 51 / 120   # of tools reaching their 200-240 min end of life, the share that failed in service


def normalised_random_walk(rng: np.random.Generator, n: int, sd: float) -> np.ndarray:
    walk = np.cumsum(rng.normal(0.0, 1.0, n))
    return (walk - walk.mean()) / walk.std() * sd


def generate(rows: int = 10_000, seed: int = 2020) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=rows, p=[0.5, 0.3, 0.2])

    air = 300.0 + normalised_random_walk(rng, rows, 2.0)
    process = 310.0 + 0.6 * (air - 300.0) + normalised_random_walk(rng, rows, 0.9)

    torque = np.clip(rng.normal(40.0, 10.0, rows), 3.8, 76.6)
    # Speed follows the power level, so high torque means low speed: rpm = A / (torque + c) with 5% noise.
    rpm = 149_900 / (torque + 57.4) * rng.lognormal(0.0, 0.05, rows)
    rpm = np.clip(rpm, 1168, 2886)

    wear = np.zeros(rows, dtype=int)
    twf = np.zeros(rows, dtype=int)
    current, end_of_life = 0, rng.integers(200, 241)
    for i, t in enumerate(types):
        current += WEAR_PER_PROCESS[t]
        if current >= end_of_life:
            if rng.random() < TOOL_FAIL_SHARE:
                twf[i] = 1                                   # tool failed in service at this process
            wear[i] = current
            current, end_of_life = 0, rng.integers(200, 241)  # new tool
        else:
            wear[i] = current

    air, process = np.round(air, 1), np.round(process, 1)
    rpm, torque = np.round(rpm).astype(int), np.round(torque, 1)
    power = torque * rpm * 2 * np.pi / 60
    hdf = (((process - air) < 8.6) & (rpm < 1380)).astype(int)
    pwf = ((power < 3500) | (power > 9000)).astype(int)
    osf = (wear * torque > np.vectorize(OSF_LIMIT.get)(types)).astype(int)
    rnf = (rng.random(rows) < 0.001).astype(int)
    failure = np.maximum.reduce([twf, hdf, pwf, osf, rnf])

    serials = rng.choice(np.arange(10_000, 100_000), size=rows, replace=False)
    return pd.DataFrame({
        "UDI": np.arange(1, rows + 1),
        "Product ID": [f"{t}{s}" for t, s in zip(types, serials)],
        "Type": types,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Machine failure": failure,
        "TWF": twf, "HDF": hdf, "PWF": pwf, "OSF": osf, "RNF": rnf,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default="data/ai4i2020_sample.csv")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()
    sample = generate(args.rows, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    modes = sample[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum().to_dict()
    print(f"Wrote {len(sample):,} rows to {args.output}; failure rate {sample['Machine failure'].mean():.2%}; modes {modes}")
