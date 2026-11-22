# 张量与参数参考

[项目首页](../README.md) · [结构流的设计](DESIGN.md)

入口是 [AlgebraModel](../model/algebra.py)。下表中，`B` 为批大小，`T` 为序列长度，`w` 为主干宽度，`k` 为矩阵边长，`V` 为词表大小。

## 前向接口

| 量 | 形状 | 说明 |
| --- | --- | --- |
| `events` | `(B, T, n_fields)` | 建议使用 `torch.long`，每个字段的索引都在 `[0, V)`；`n_fields` 默认 4。 |
| `h` | `(B, T, w)` | 各字段独立嵌入后相加，逐层经过语义块。 |
| `dₜ`、`aₜ`、`Aₜ`、`Zₜ` | 每个位置 `(B, k, k)` | 输入驱动、无迹状态、当前变换及有序乘积；`nocompose` 的读出直接用 `Aₜ`。 |
| `logits` | `(B, T, V)` | 每个位置经 `LayerNorm` 和输出线性层；未做 softmax。 |

[LocalBlock](../model/block.py) 的注意力头宽为 64，`width` 必须是 64 的倍数。`window` 含当前位置，默认值为 8。模型按编号从 0 开始遍历语义块，在 `state_layers` 指定的位置紧接一个结构层；例如默认的 `(1, 3)` 表示第 2、4 个块之后。每次调用 `forward` 都从头初始化结构状态，不跨调用保存前缀。

## 一个结构层的参数

| 参数 | 存储形状 | `w=192, k=3` 时的数量 |
| --- | --- | ---: |
| `raw_v`、`d`、`omega` | `(3,)`、`(2,)`、`(2,)` | 7 |
| `to_drive.weight`、`to_drive.bias` | `(k², w)`、`(k²,)` | 1,737 |
| `out.weight` | `(w, k²)`，无偏置 | 1,728 |
| 合计 | `7 + 2wk² + k²` | 3,472 |

不同结构层各有一套以上参数。`model.generators()` 返回以层编号字符串为键的字典，每项包含 `v`（长度 3）、`d₀`、`d₁`、`ω₀` 和 `ω₁`。`v` 是 `raw_v` 经过 softplus 后的值。

## 实现索引

以下方法均位于 [model/algebra.py](../model/algebra.py)。

| 方法 | 输入与输出 | 实现要点 |
| --- | --- | --- |
| `AlgebraState.grad_potential` | `a → ∇V` | `∇V = 2v₂ aᵀ + 2v₃ a + 4v₄(Ca − aC)`，其中 `C = aaᵀ − aᵀa`。 |
| `AlgebraState.flow` | `a, drive → Δa` | 在负梯度上叠加对称、反对称分量及位置驱动；时间步长与无迹投影在调用方应用。 |
| `AlgebraState.decompose` | `a → R, P` | 返回正交因子和对称收缩因子，调用方按 `R @ P` 合成当前变换。 |
| `AlgebraState._prefix_product` | `(B, T, k, k) → 同形张量` | 第 t 项存放到该位置为止的左乘积；以倍增步长计算所有前缀。 |
| `AlgebraState._stream` | `h → h + readout` | `out` 作用于展平的矩阵，其输出宽度为 `w`。 |

`softplus` 保证势函数的三个系数为正，但这本身不足以保证整个势有下界。例如对反对称的 `a`，有 `tr(a²) = −‖a‖²_F` 且 `C = 0`，二次项的符号取决于 `v₃ − v₂`。设计文档中的谱范数界来自 `R`、`P` 的构造，不需要假设 `V` 有下界。

## 可选统计量

向 `model(events, stats=stats)` 传入空字典即可收集。代码在每个启用的结构层写入相同的键；启用多层时，最终保留最后一层的数据。`arm='off'` 不写这些键。

| 键 | 计算方式 |
| --- | --- |
| `a_var` | 对 `a` 沿时间维取样本方差，再对批和矩阵元素取平均。`T = 1` 时样本方差无定义。 |
| `z_norm` | `Zₜ` 的 Frobenius 范数，对批和所有位置取平均。 |
| `z_last` | 末位置 `Z` 的 Frobenius 范数，对批取平均。 |
| `mat_norm` | 每个样本先取时间上最大的 `Aₜ` 的 Frobenius 范数，再对批取平均。 |

这里的 `mat_norm` 不是谱范数；例如 `k × k` 单位矩阵的谱范数是 1，Frobenius 范数是 `√k`。检查设计中的谱范数界应显式使用 `torch.linalg.matrix_norm(A, ord=2)`。
