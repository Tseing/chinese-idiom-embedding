import pickle
from typing import List

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy import ndarray
from numpy.typing import NDArray
from sklearn.metrics.pairwise import euclidean_distances

from src.chinese_idiom.model import ModelLabel
from src.chinese_idiom.utils import batch_embedding


def gen_text_vectors(model_label: ModelLabel):
    # optimizing scripts speed
    import torch
    import torch_npu

    device = torch.device(f"npu:0")
    torch.npu.set_device(device)
    texts = pd.read_csv("data/test_input.csv", chunksize=chunk_size)
    text_vectors = batch_embedding(model_label, texts, device)
    pickle.dump(text_vectors, open(f"output/{model_label.value}_text_vectors.pkl", "wb"))


def match_idioms(
    text_vectors: Float[ndarray, "B D"],
    idiom_vectors: Float[ndarray, "V D"],
    idioms: List[str],
    topk: int = 1,
) -> List[NDArray]:
    """find the most similar words with embedding vectors.

    Parameters
    ----------
    text_vectors : Float[ndarray, &quot;B D&quot;]
        B: batch size, D: embedding dimension
    idiom_vectors : Float[ndarray, &quot;V D&quot;]
        V: vocab size, D: embedding dimension
    """
    assert (
        len(idioms) == idiom_vectors.shape[0]
    ), f"Unmatched shape: `idoms` {len(idioms)} and `idiom_vectors` {idiom_vectors.shape}"
    idioms = np.array(idioms)
    scores: Float[ndarray, "B V"] = euclidean_distances(text_vectors, idiom_vectors)
    matched_idx = np.argsort(scores, axis=1)[:, :topk]

    return [idioms[idx] for idx in matched_idx]


if __name__ == "__main__":
    # Config
    model_label = ModelLabel.Xiaobu
    chunk_size = 256
    topk = 5

    gen_text_vectors(model_label)

    idioms = pd.read_csv("data/idioms.csv")["word"].to_list()
    idiom_vectors = pickle.load(open(f"output/{model_label.value}_vectors.pkl", "rb"))
    text_vectors = pickle.load(open(f"output/{model_label.value}_text_vectors.pkl", "rb"))

    res = pd.DataFrame(match_idioms(text_vectors, idiom_vectors, idioms, topk))
    res.to_csv(
        f"output/{model_label.value}_top{topk}_results.csv", sep=" ", header=False, index=False
    )
    print(f"Done. '{model_label.value}_top{topk}_results.csv' is generated.")
