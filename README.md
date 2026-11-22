# Algebraic Structure Stream

Algebraic Structure Stream 是一个用 PyTorch 实现的因果序列模型。它在局部注意力层之间插入一条矩阵递推：每个位置生成一个变换，按输入顺序左乘到已有乘积上。处理完第 3 个位置时，交给主干的是 `A₃ A₂ A₁` 的读出。一般情况下，交换其中两个因子就会改变读出，即使这三个矩阵本身都没有变。

[设计说明](docs/DESIGN.md) · [张量与参数参考](docs/MODEL_MAP.md) · [前向示例](examples/forward.py)

## 在 CPU 上运行

在安装了 Python 3.12 的环境中执行：

```text
git clone https://github.com/Braun2009/algstruct.git
cd algstruct
python -m pip install -r requirements.txt
python -m examples.forward
```

示例随机初始化一个小模型，输入两条各含 8 个事件的序列。每个事件有 4 个离散字段，每个字段的取值范围都是 0 到 31。无需下载数据或权重。

```text
arm=full
events: (2, 8, 4)
logits: (2, 8, 32)
```

在自己的程序里也可以直接从仓库根目录导入 `model`：

```text
import torch
from model import AlgebraConfig, AlgebraModel

cfg = AlgebraConfig(width=64, layers=2, state_layers=(1,))
model = AlgebraModel(cfg, vocab=32).eval()
events = torch.randint(32, (2, 8, 4))
with torch.inference_mode():
    logits = model(events)  # 每个位置输出 32 个分数
```

## 一次前向如何工作

四个字段各有一张嵌入表。模型将同一位置的嵌入相加，再用因果注意力和 MLP 得到语义向量 `h`。在 `state_layers` 指定的层，结构流维护三个不同的量：

| 量 | 含义 | 在时间步 t 的用途 |
| --- | --- | --- |
| `aₜ` | 无迹矩阵状态 | 由上一步状态和当前 `hₜ` 更新，决定本步变换。 |
| `Aₜ` | 本步变换 | 由 `aₜ` 构造，可以旋转或收缩不同方向。 |
| `Zₜ` | 从序列起点至今的乘积 | 从单位矩阵出发，按 `Zₜ = Aₜ Zₜ₋₁` 更新；展平后经线性层加回 `hₜ`。 |

这个构造给出一个明确的界：精确算术下，`Aₜ` 和 `Zₜ` 的谱范数都不超过 1。它只限制结构流的放大倍数；长序列中的信号仍然可能衰减。[设计说明](docs/DESIGN.md) 给出了一个两步就发生衰减的例子。

## 比较四种配置

用 `python -m examples.forward --arm nocompose` 可以切换配置。写自己的调用时，设置 `AlgebraConfig(arm=...)` 即可。

| `arm` | 读出所用的矩阵 | 比较的因素 |
| --- | --- | --- |
| `full` | 截至当前位置的有序乘积 | 完整结构流 |
| `nocompose` | 仅当前的 `Aₜ` | 去掉变换的累积；生成 `Aₜ` 的状态仍按时间递推 |
| `nodrive` | 从零初态得到的单位矩阵 | 切断语义输入对结构状态的驱动，保留读出层 |
| `off` | 无结构读出 | 只运行语义主干 |

仓库提供架构和前向示例，不包含预训练权重、训练数据或任务评测结果。示例的输出用于核对接口，不能据此比较四种配置的任务表现。

许可：[Apache-2.0](LICENSE)。DeepSeek-Flash 参与了架构实现。
