import torch
import torch_npu
from sentence_transformers import SentenceTransformer

device = torch.device(f"npu:0")
torch.npu.set_device(device)
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer("../models/xiaobu-embedding-v2", device=device)
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)

similarity = embeddings_1 @ embeddings_2.T
print(similarity)
