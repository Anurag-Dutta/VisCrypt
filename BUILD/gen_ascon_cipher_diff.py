import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

WORD_BITS = 64
STATE_WORDS = 5
STATE_BITS = 320
MASK64 = (1 << 64) - 1

DELTA = np.array([0, 0, 0, 0, 0x010], dtype=np.uint64)
SAVE_DIR = f"/data1/home/virendra/lwc/ascon/indiff{format(int(DELTA[-1]), 'x')}"
os.makedirs(SAVE_DIR, exist_ok=True)

ROUND_CONST = np.array(
    [
        0xF0, 0xE1, 0xD2, 0xC3,
        0xB4, 0xA5, 0x96, 0x87,
        0x78, 0x69, 0x5A, 0x4B,
        0x3C, 0x2D, 0x1E, 0x0F,
    ],
    dtype=np.uint64,
)

def rot(x, r):
    return ((x >> r) | (x << (64 - r))) & MASK64

def add_round_constant(S, r, R):
    S[2] ^= ROUND_CONST[12 - R + r]

def sbox(S):
    S[0] ^= S[4]
    S[4] ^= S[3]
    S[2] ^= S[1]

    T = (~S) & MASK64
    T[0] &= S[1]
    T[1] &= S[2]
    T[2] &= S[3]
    T[3] &= S[4]
    T[4] &= S[0]

    S[0] ^= T[1]
    S[1] ^= T[2]
    S[2] ^= T[3]
    S[3] ^= T[4]
    S[4] ^= T[0]

    S[1] ^= S[0]
    S[0] ^= S[4]
    S[3] ^= S[2]
    S[2] = (~S[2]) & MASK64

def linear_layer(S):
    S[0] ^= rot(S[0], 19) ^ rot(S[0], 28)
    S[1] ^= rot(S[1], 61) ^ rot(S[1], 39)
    S[2] ^= rot(S[2], 1) ^ rot(S[2], 6)
    S[3] ^= rot(S[3], 10) ^ rot(S[3], 17)
    S[4] ^= rot(S[4], 7) ^ rot(S[4], 41)

def ascon_permutation(S, R):
    for r in range(R):
        add_round_constant(S, r, R)
        sbox(S)
        linear_layer(S)
    return S

def state_to_bits(X):
    return np.unpackbits(
        X.view(np.uint8).reshape(-1, 8)[:, ::-1],
        axis=1,
    ).reshape(len(X), STATE_BITS)

def generate_dataset(d, R):
    X_data = []
    Y_data = []

    N = 2 ** d

    for _ in range(N):
        p0 = np.random.randint(0, 2**64, size=5, dtype=np.uint64)
        p1 = np.random.randint(0, 2**64, size=5, dtype=np.uint64)
        p2 = p1 ^ DELTA

        c0 = ascon_permutation(p0.copy(), R)
        c1 = ascon_permutation(p1.copy(), R)
        c2 = ascon_permutation(p2.copy(), R)

        x0 = c0 ^ c1
        x1 = c1 ^ c2

        X_data.append(x0)
        Y_data.append(0)
        X_data.append(x1)
        Y_data.append(1)

    return (
        np.array(X_data, dtype=np.uint64),
        np.array(Y_data, dtype=np.uint8),
    )

def save_ascon_dataset(d, R, train_size, test_size):
    X_state, Y = generate_dataset(d, R)
    X_bits = state_to_bits(X_state)

    columns = [f"bit_{i}" for i in range(STATE_BITS)]
    df = pd.DataFrame(X_bits, columns=columns)
    df["label"] = Y

    Xtr, Xte, ytr, yte = train_test_split(
        df[columns],
        df["label"],
        train_size=train_size,
        test_size=test_size,
        stratify=df["label"],
        random_state=42,
    )

    train_df = Xtr.copy()
    train_df["label"] = ytr.values

    test_df = Xte.copy()
    test_df["label"] = yte.values

    train_path = os.path.join(SAVE_DIR, f"ascon_train_round{R}.csv")
    test_path = os.path.join(SAVE_DIR, f"ascon_test_round{R}.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"saved ascon_train_round{R}.csv")
    print(f"saved ascon_test_round{R}.csv")
    print("train labels:", np.bincount(ytr))
    print("test labels :", np.bincount(yte))

if __name__ == "__main__":
    d = 20
    train_size = 1_000_000
    test_size = 40_000

    for R in [1, 2, 3, 4, 5, 6]:
        print(f"generating round {R}")
        save_ascon_dataset(d, R, train_size, test_size)

    print("all ASCON datasets generated")