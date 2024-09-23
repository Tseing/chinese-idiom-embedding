import torch
import torch.nn.functional as F
import torch_npu
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, 'summit define')
]

# documents = [
#     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#     "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
# ]
input_texts = queries

device = torch.device(f"npu:0")
torch.npu.set_device(device)

print(input_texts)

tokenizer = AutoTokenizer.from_pretrained("../models/gte-Qwen2-7B-instruct")
model = AutoModel.from_pretrained("../models/gte-Qwen2-7B-instruct", device_map=device)

max_length = 512

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
).to(device)
outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings)
print(embeddings.shape)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
