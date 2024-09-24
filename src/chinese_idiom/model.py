import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import torch
import torch_npu
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from torch.functional import F
from transformers import AutoModel, AutoTokenizer


class ModelLabel(Enum):
    PuffBase = "puff-base-v1"
    Tao = "tao-8k"
    GteLargeZh = "gte-large-zh"
    GteQwen2 = "gte-Qwen2-7B-instruct"
    Conan = "Conan-embedding-v1"
    Xiaobu = "xiaobu-embedding-v2"


class EmbeddingModel(ABC):
    @abstractmethod
    def __call__(self, input_texts: List[str]) -> NDArray:
        raise NotImplementedError


class PuffEmbedding(EmbeddingModel):
    def __init__(self, model_dir: str, vector_dim: int, device) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_dir, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        vector_linear = torch.nn.Linear(
            in_features=self.model.config.hidden_size, out_features=vector_dim
        )
        vector_linear_dict = {
            k.replace("linear.", ""): v
            for k, v in torch.load(
                os.path.join(model_dir, f"2_Dense_{vector_dim}/pytorch_model.bin"),
                map_location=device,
            ).items()
        }
        vector_linear.load_state_dict(vector_linear_dict)
        self.vector_linear = vector_linear.to(device)
        self.device = device

    def __call__(self, input_texts: List[str]) -> NDArray:
        with torch.no_grad():
            inputs = self.tokenizer(
                input_texts, padding="longest", truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            attention_mask = inputs["attention_mask"]
            last_hidden_state = self.model(**inputs)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            vectors = normalize(self.vector_linear(vectors).cpu().numpy())

        return vectors


class GeneralEmbedding(EmbeddingModel):
    def __init__(self, model_dir: str, device: torch.device) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_dir, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = device

    def __call__(self, input_texts: List[str]) -> NDArray:
        with torch.no_grad():
            inputs = self.tokenizer(
                input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            vectors = outputs.last_hidden_state[:, 0]

            vectors = F.normalize(vectors, p=2, dim=1).cpu().numpy()

        return vectors


class GteQwenEmbedding(EmbeddingModel):
    def __init__(self, model_dir: str, device: torch.device) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_dir, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = device

    def __call__(self, input_texts: List[str]) -> NDArray:
        with torch.no_grad():
            inputs = self.tokenizer(
                input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            vectors = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])

            vectors = F.normalize(vectors, p=2, dim=1).cpu().numpy()

        return vectors

    @staticmethod
    def last_token_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
            ]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"


class XiaobuEmebdding(EmbeddingModel):
    def __init__(self, model_dir: str, device: torch.device) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_dir, device=device)
        self.device = device

    def __call__(self, input_texts: List[str]) -> NDArray:
        vectors = self.model.encode(input_texts, normalize_embeddings=True)

        return vectors
