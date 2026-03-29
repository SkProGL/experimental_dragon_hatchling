import numpy as np
import os
import pandas as pd


_dataset_bytes = None


def load_tinystories():
    base_path = os.path.join(os.path.dirname(__file__), "tinystories")

    global _dataset_bytes
    if _dataset_bytes is not None:
        return _dataset_bytes

    # ---- TRAIN ----
    train_files = [
        "train-00000.parquet",
        "train-00001.parquet",
        "train-00002.parquet",
        "train-00003.parquet",
    ]

    dfs = [pd.read_parquet(os.path.join(base_path, f)) for f in train_files]
    train_df = pd.concat(dfs, ignore_index=True)

    train_texts = train_df["text"].astype(str).tolist()
    train_full = "\n\n<doc>\n\n".join(train_texts)

    train_bytes = np.frombuffer(train_full.encode("utf-8"), dtype=np.uint8)

    # ---- VALIDATION ----
    val_path = os.path.join(base_path, "validation-00000.parquet")
    val_df = pd.read_parquet(val_path)

    val_texts = val_df["text"].astype(str).tolist()
    val_full = "\n\n<doc>\n\n".join(val_texts)

    val_bytes = np.frombuffer(val_full.encode("utf-8"), dtype=np.uint8)

    _dataset_bytes = {"train": train_bytes, "val": val_bytes, }

    return _dataset_bytes


def load_wiki():
    WIKI_PATH = os.path.join(os.path.dirname(
        __file__), "simple-wikipedia.parquet")

    global _dataset_bytes
    if _dataset_bytes is not None:
        return _dataset_bytes

    df = pd.read_parquet(WIKI_PATH)

    texts = df["text"].astype(str).tolist()

    # marks document boundaries, helps model learn
    full_text = "\n\n<doc>\n\n".join(texts)

    # converts string into utf-8 encoded raw bytes,
    # then converts to byte-level integer array (0-255)
    _dataset_bytes = np.frombuffer(
        full_text.encode("utf-8"),
        dtype=np.uint8,
    )
    return _dataset_bytes
