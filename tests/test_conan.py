import torch
import torch.nn.functional as F
import torch_npu
from transformers import AutoModel, AutoTokenizer

input_texts = ["中国的首都是哪里", "你喜欢去哪里旅游", "北京", "今天中午吃什么"]

device = torch.device(f"npu:0")
torch.npu.set_device(device)

tokenizer = AutoTokenizer.from_pretrained("../models/Conan-embedding-v1")
model = AutoModel.from_pretrained("../models/Conan-embedding-v1", device_map=device)

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
).to(device)

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]

# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings.shape)
print(embeddings[:, :4])
