import os
import numpy as np
import pandas as pd
from tqdm import tqdm

WORD_SIZE = 16
MASK_VAL = (1 << WORD_SIZE) - 1
ALPHA = 7
BETA = 2
DELTA = 0x0010

delta_hex = format(DELTA, "x")
SAVE_DIR = f"/speck3264/indiff{delta_hex}"
os.makedirs(SAVE_DIR, exist_ok=True)

def rotate_right(x, r):
    return ((x >> r) | (x << (WORD_SIZE - r))) & MASK_VAL

def rotate_left(x, r):
    return ((x << r) & MASK_VAL) | (x >> (WORD_SIZE - r))

def speck_round(x, y, k):
    x = rotate_right(x, ALPHA)
    x = (x + y) & MASK_VAL
    x ^= k
    y = rotate_left(y, BETA)
    y ^= x
    return x, y

def speck_encrypt(block, subkeys):
    x = (block >> WORD_SIZE) & MASK_VAL
    y = block & MASK_VAL
    for k in subkeys:
        x, y = speck_round(x, y, k)
    return (x << WORD_SIZE) | y

def expand_key(key, rounds):
    words = [(key >> (16 * i)) & MASK_VAL for i in reversed(range(4))]
    k = words[0]
    l = words[1:]
    subkeys = [k]
    for i in range(rounds - 1):
        li = rotate_right(l[i], ALPHA)
        li = (li + subkeys[i]) & MASK_VAL
        li ^= i
        k = rotate_left(subkeys[i], BETA) ^ li
        subkeys.append(k)
        l.append(li)
    return subkeys

class SpeckDatasetGenerator:
    def __init__(self, delta, seed):
        self.delta = delta
        self.seed = seed
        np.random.seed(seed)

    def _int_to_bits(self, value):
        return np.unpackbits(
            np.array(
                [
                    (value >> 24) & 0xFF,
                    (value >> 16) & 0xFF,
                    (value >> 8) & 0xFF,
                    value & 0xFF,
                ],
                dtype=np.uint8,
            ),
            bitorder="big",
        ).astype(np.float32)

    def make_speck_dataset(self, n_pairs, rounds):
        n = 2 * n_pairs
        X = np.zeros((n, WORD_SIZE * 2), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)

        for i in tqdm(range(n_pairs), desc=f"rounds={rounds}"):
            key = int.from_bytes(os.urandom(8), "big")
            subkeys = expand_key(key, rounds)

            p0 = int.from_bytes(os.urandom(4), "big")
            p1 = p0 ^ self.delta

            c0 = speck_encrypt((p0 << 16) | (p0 >> 16), subkeys)
            c1 = speck_encrypt((p1 << 16) | (p1 >> 16), subkeys)

            diff_real = c0 ^ c1
            X[2 * i] = self._int_to_bits(diff_real)
            y[2 * i] = 1

            r = int.from_bytes(os.urandom(4), "big")
            rc = speck_encrypt((r << 16) | (r >> 16), subkeys)

            diff_rand = rc ^ c0
            X[2 * i + 1] = self._int_to_bits(diff_rand)
            y[2 * i + 1] = 0

        return X, y

    def generate_train_test_split(self, rounds, train_pairs, test_pairs):
        train_X, train_y = self.make_speck_dataset(train_pairs, rounds)

        old_seed = self.seed
        np.random.seed(self.seed + rounds * 10000)
        test_X, test_y = self.make_speck_dataset(test_pairs, rounds)
        np.random.seed(old_seed)

        return train_X, train_y, test_X, test_y

    def save_to_csv(self, X, y, path):
        cols = [f"bit_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df["label"] = y
        df.to_csv(path, index=False)

    def generate_all_rounds(self, rounds_list, train_pairs, test_pairs):
        results = []

        for r in rounds_list:
            train_X, train_y, test_X, test_y = self.generate_train_test_split(
                r, train_pairs, test_pairs
            )

            train_path = os.path.join(SAVE_DIR, f"speck_train_round_{r}.csv")
            test_path = os.path.join(SAVE_DIR, f"speck_test_round_{r}.csv")

            self.save_to_csv(train_X, train_y, train_path)
            self.save_to_csv(test_X, test_y, test_path)

            results.append(
                (
                    r,
                    len(train_y),
                    len(test_y),
                    np.mean(train_y),
                    np.mean(test_y),
                )
            )

        for r, tr_n, te_n, tr_b, te_b in results:
            print(r, tr_n, te_n, f"{tr_b:.3f}", f"{te_b:.3f}")

def main():
    generator = SpeckDatasetGenerator(delta=DELTA, seed=42)
    generator.generate_all_rounds(
        rounds_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        train_pairs=500000,
        test_pairs=20000,
    )

if __name__ == "__main__":
    main()