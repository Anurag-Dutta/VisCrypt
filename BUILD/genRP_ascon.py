import os
import numpy as np
import pandas as pd
from pyts.image import RecurrencePlot
from PIL import Image
from tqdm import tqdm

ROUNDS = 6
CSV_PATTERN = "ascon_{split}_round{r}.csv"
OUTPUT_DIR_PATTERN = "round_{r}_{split}_rp"

SAVE_DIR = "/ascon/indiff40"
os.makedirs(SAVE_DIR, exist_ok=True)

USE_FIXED_THRESHOLD = True
EPSILON = 0.1
USE_PERCENTAGE = False
PERCENTAGE = 10

if USE_FIXED_THRESHOLD:
    rp = RecurrencePlot(
        dimension=1,
        time_delay=1,
        threshold=EPSILON,
        flatten=False,
    )
elif USE_PERCENTAGE:
    rp = RecurrencePlot(
        dimension=1,
        time_delay=1,
        threshold="point",
        percentage=PERCENTAGE,
        flatten=False,
    )
else:
    rp = RecurrencePlot(
        dimension=1,
        time_delay=1,
        threshold=None,
        flatten=False,
    )

for r in range(1, ROUNDS + 1):
    for split in ("train", "test"):
        csv_path = os.path.join(
            SAVE_DIR,
            CSV_PATTERN.format(split=split, r=r),
        )

        if not os.path.exists(csv_path):
            print(f"{csv_path} not found")
            continue

        out_dir = os.path.join(
            SAVE_DIR,
            OUTPUT_DIR_PATTERN.format(r=r, split=split),
        )

        for lbl in ("0", "1"):
            os.makedirs(os.path.join(out_dir, lbl), exist_ok=True)

        df = pd.read_csv(csv_path)
        X = df.drop("label", axis=1).values
        y = df["label"].values.astype(int)

        print(f"round {r} {split} {len(X)} samples")

        for i in tqdm(range(len(X)), desc=f"r{r}-{split}"):
            ts = X[i].reshape(1, -1)
            mat = rp.fit_transform(ts)[0]
            img = (mat * 255).astype(np.uint8)

            label = str(y[i])
            name = f"{i:06d}.png"
            path = os.path.join(out_dir, label, name)

            Image.fromarray(img).save(path)