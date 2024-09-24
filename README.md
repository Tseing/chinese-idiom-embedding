# chinese-idiom-embedding

测试环境：

```yaml
OS: EulerOS 2.0 (SP8)
NPU: 910B (23.0.rc2)
CANN: CANN-7.0.0.alpha003
Python: 3.9.18
PyTorch: 2.1.0
```

| Model                                                                                         | Model Family | Embedding Dimension | Max Length | Latest Update | Version                                    |
| --------------------------------------------------------------------------------------------- | ------------ | ------------------- | ---------- | ------------- | ------------------------------------------ |
| [Amu/tao-8k](https://huggingface.co/Amu/tao-8k)                                               | stella-v2    | 1024                | 512        | 23 Dec 3      | `bc783a14a907aa520ebf978787d3d6513803de13` |
| [infgrad/puff-base-v1](https://huggingface.co/infgrad/puff-base-v1)                           | puff         | 4096                | 512        | 24 Apr 6      | `60b9876024e8163191923a73639d31d1381cec00` |
| [thenlper/gte-large-zh](https://huggingface.co/thenlper/gte-large-zh)                         | GTE          | 1024                | 512        | 24 Feb 5      | `64c364e579de308104a9b2c170ca009502f4f545` |
| [Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) | GTE & Qwen2  | 3584                | 512        | 24 Aug 9      | `f47e3b5071bf9902456c6cbf9b48b59994689ac0` |
| [TencentBAC/Conan-embedding-v1](https://huggingface.co/TencentBAC/Conan-embedding-v1)         | Conan        | 1024                | 512        | 24 Sep 6      | `fbdfbc53cd9eff1eb55eadc28d99a9d4bff4135f` |
| [lier007/xiaobu-embedding-v2](https://huggingface.co/lier007/xiaobu-embedding-v2)             | piccolo      |                     |            | 24 Jun 30     | `ee0b4ecdf5eb449e8240f2e3de2e10eeae877691` |

## Dataset

- `idioms.csv`（30895）：数据来自 [crazywhalecc/idiom-database](https://github.com/crazywhalecc/idiom-database)，包含成语词条与释义，预处理过程中清洗了其中的特殊符号和漏字。
- `idiom_subset.csv`（200）：从 `idioms.csv` 中随机选取 200 条。
- `test_input.csv`（2972）：由科大讯飞提供，仅包含成语释义，没有对应的成语词条。

## Recall Top@5

| Model                 | Recall  |
| --------------------- | ------- |
| tao-8k                | 0.66521 |
| puff-base-v1          | 0.70222 |
| gte-large-zh          | 0.52355 |
| get-Qwen2-7B-instruct | 0.17968 |
| Conan-embedding-v1    | 0.48284 |
| xiaobu-embedding-v2   | 0.49092 |

## Embedding Visualization

使用 UMAP 对所有成语文本的向量可视化，得到全局结构。另随机选取 200 条成语文本，展示向量的局部分布。随机选取 20 条测试文本，展示了最邻近 5 个向量的分布。代码见 [Jupyter Notebook](./scripts/plots.ipynb)。

### Distribution of idioms

### Distribution of idioms subset

### Distribution of matched results

#### tao-8k

![tao-8k_test](./images/tao-8k_test.png)

#### puff-base-v1

![puff-base-v1_test](./images/puff-base-v1_test.png)

#### gte-large-zh

![gte-large-zh_test](./images/gte-large-zh_test.png)

#### gte-Qwen2-7B-instruct

#### Conan-embedding-v1

#### xiaobu-embedding-v2
