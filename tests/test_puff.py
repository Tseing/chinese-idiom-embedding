import os

import torch
import torch_npu
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

# 待编码文本
texts = ["通用向量编码", "hello world", "支持中英互搜，不建议纯英文场景使用"]
# 模型目录
model_dir = "../models/puff-base-v1"

device = torch.device(f"npu:0")
torch.npu.set_device(device)

vector_dim = 4096
model = AutoModel.from_pretrained(model_dir).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir)
vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
vector_linear_dict = {
    k.replace("linear.", ""): v
    for k, v in torch.load(
        os.path.join(model_dir, f"2_Dense_{vector_dim}/pytorch_model.bin"), map_location=device
    ).items()
}
vector_linear.load_state_dict(vector_linear_dict)
with torch.no_grad():
    input_data = tokenizer(
        texts, padding="longest", truncation=True, max_length=512, return_tensors="pt"
    )
    attention_mask = input_data["attention_mask"]
    last_hidden_state = model(**input_data)[0]
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    vectors = normalize(vector_linear(vectors).cpu().numpy())
print(vectors.shape)
print(vectors[:, :4])
