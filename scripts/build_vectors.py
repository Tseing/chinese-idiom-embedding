import pickle
from typing import Iterable

import pandas as pd
import torch
import torch_npu

from src.chinese_idiom.model import ModelLabel
from src.chinese_idiom.utils import batch_embedding


def gen_vectors(
    model_label: ModelLabel,
    chunks: Iterable[pd.DataFrame],
    device: torch.device,
    instruct: bool = False,
):
    vectors = batch_embedding(model_label, chunks, device, instruct)
    pickle.dump(vectors, open(f"output/{model_label.value}_vectors.pkl", "wb"))


if __name__ == "__main__":
    device = torch.device(f"npu:0")
    torch.npu.set_device(device)

    model_label = ModelLabel.GteQwen2
    instruct = True
    chunk_size = 16

    print(f"EmbeddingModel: {model_label.value}, chunksize: {chunk_size}.")
    chunks = pd.read_csv("data/idioms.csv", chunksize=chunk_size)
    gen_vectors(model_label, chunks, device, instruct)
