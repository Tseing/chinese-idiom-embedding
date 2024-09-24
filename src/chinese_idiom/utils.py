import os.path as osp
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from .model import (
    EmbeddingModel,
    GeneralEmbedding,
    ModelLabel,
    PuffEmbedding,
    GteQwenEmbedding,
    XiaobuEmebdding,
)


TASK_INSTRUCT = "给定一个描述，告诉我意思最接近的成语"


def batch_embedding(
    model_label: ModelLabel,
    chunks: Iterable[pd.DataFrame],
    device: torch.device,
    instruct: bool = False,
) -> NDArray:
    model: EmbeddingModel
    model_dir = osp.join("../models", model_label.value)
    if model_label is ModelLabel.PuffBase:
        model = PuffEmbedding(model_dir, 4096, device)
    elif model_label is ModelLabel.GteQwen2:
        model = GteQwenEmbedding(model_dir, device)
    elif model_label is ModelLabel.Xiaobu:
        model = XiaobuEmebdding(model_dir, device)
    elif (
        model_label is ModelLabel.Tao
        or model_label is ModelLabel.GteLargeZh
        or model_label is ModelLabel.Conan
    ):
        model = GeneralEmbedding(model_dir, device)

    else:
        assert False, f"Unsupported model: {model_label}."

    vectors: List[NDArray] = []
    for chunk in tqdm(chunks):
        texts = chunk["explanation"].to_list()

        if instruct and model_label is ModelLabel.GteQwen2:
            texts = [GteQwenEmbedding.get_detailed_instruct(TASK_INSTRUCT, text) for text in texts]

        vectors.append(model(texts))

    return np.concatenate(vectors, axis=0)
