import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

input_texts = [
    "中国的首都是哪里",
    "你喜欢去哪里旅游",
    "北京",
    "今天中午吃什么"
]

tokenizer = AutoTokenizer.from_pretrained("../models/tao-8k")
model = AutoModel.from_pretrained("../models/tao-8k")

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]
 
# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings.shape)
print(embeddings[:, :4])