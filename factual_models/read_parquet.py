import pandas as pd
import numpy as np
import os


def load_wiki_bytes():
    WIKI_PATH = os.path.join(os.path.dirname(
        __file__), "simple-wikipedia.parquet")
    _wiki_bytes = None
    print(WIKI_PATH)

    df = pd.read_parquet(WIKI_PATH)
    print(df)

    texts = df["text"].astype(str).tolist()
    print("\n\n".join(texts[:2]))
    # print(texts)

    # marks document boundaries, helps model learn
    full_text = "\n\n<doc>\n\n".join(texts)

    # converts string into utf-8 encoded raw bytes,
    # then converts to byte-level integer array (0-255)
    _wiki_bytes = np.frombuffer(
        full_text.encode("utf-8"),
        dtype=np.uint8,
    )
    return _wiki_bytes


load_wiki_bytes()
