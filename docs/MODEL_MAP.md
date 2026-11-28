# 张量与参数参考

[项目首页](../README.md) · [思想](PHILOSOPHY.md) · [设计](DESIGN.md) · [实测](EVIDENCE.md) · [与相关工作的分野](RELATED.md)



## 摘要

逐张量、逐参数的实现映射：每个形状的含义、每个参数的数量与初始化的物理依据、
每个方法的输入输出。代码与本文不一致时，以本文为准并修改代码。

**关键词**：张量形状 · 参数表 · 初始化 · 统计量 · 不变量检查

入口是 [AlgebraModel](../model/algebra.py)。下表中 `B` 批大小、`T` 序列长度、`w` 语义宽度、
`k` 矩阵边长、`q` 每分量带宽、`V` 词表大小。结构空间宽度 `m = k·q`，与 `w` 无关。

## 前向接口

| 量 | 形状 | 说明 |
| --- | --- | --- |
| `events` | `(B, T, n_fields)` | 每字段索引在 `[0, V)`；`n_fields` 默认 4 |
| `h` | `(B, T, w)` | 各字段嵌入相加，逐层经过语义块 |
| `d_t` | `(B, T, k, k)` | 驱动，源项 |
| `Δτ_t` | `(B, T, 1, 1)` | 固有时间步长，`∈ (0, dt)` |
| `a_t` `R_t` `P_t` `Z_t` | 每位置 `(B, k, k)` | 局部生成元、旋转、收缩、有序乘积 |
| `s` | `(B, T, k, q)` | 接口输出，结构作用的操作数 |
| `logits` | `(B, T, V)` | 经 `LayerNorm` 和输出线性层，未做 softmax |

## 一层的两段

语义段 [LocalBlock](../model/block.py)：注意力头宽固定 64，`width` 必须是 64 的倍数。
`window` 含当前位置，默认 `None` 表示全因果注意力。`q`、`k` 施加 [RoPE](../model/block.py)，
`v` 不施加——被平均的量不能被旋转。

结构段 [AlgebraState](../model/algebra.py)，插在 `state_layers` 指定的语义块之后：

```text
d_t   = W_d · LN(h_t)                     (B,T,k,k)
τ_t   = dt · σ(w_τ · LN(h_t))             (B,T,1,1)
a_t   ← relax(a=0, d_t) × n_relax         (B,T,k,k)   逐 token，无跨时间递推
R_t   = exp(τ_t · skew(a_t))
P_t   = exp(τ_t · (sym(a_t) − λmax I))
Z_t   = P_t · prefix(R)                   (B,T,k,k)
s     = W_s h_t                           (B,T,k,q)
h_t   ← h_t + W_o vec(Z_t s)              (B,T,w)
```

## 参数

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `gen.raw_v` | `(3,)` | `v₂ = softplus(p₀)`，`v₃ = v₂ + softplus(p₁)`，`v₄ = softplus(p₂)` |
| `gen.d` | `(2,)` | `d₀`、`d₁` |
| `gen.omega` | `(2,)` | `ω₀`、`ω₁` |
| `dn.weight`、`dn.bias` | `w`、`w` | 驱动前的 `LayerNorm`：驱动幅度必须与深度无关 |
| `to_drive.weight`、`to_drive.bias` | `(k², w)`、`(k²,)` | 语义 → 定律（源项） |
| `to_tau.weight`、`to_tau.bias` | `(1, w)`、`(1,)` | 固有时间步长，`σ` 后乘 `dt` |
| `to_s.weight` | `(k·q, w)`，无偏置 | 语义 → 结构空间（接口） |
| `out.weight` | `(w, k·q)`，无偏置 | 结构空间 → 语义（接口），小随机初始化 |

`gen` 全局一份，所有结构层共享。其余每层各一套。
`v₃ = v₂ + softplus(·)` 保证 `v₃ > v₂`，否则 `V` 在反对称子空间上恒平，
`da_skew = d_t,skew` 会变成无阻尼积分器。

`model.generators()` 返回 7 个系数的当前值。

## 实现索引

| 方法 | 输入与输出 | 要点 |
| --- | --- | --- |
| `Generator.coeffs` | `→ v, d₀, d₁, ω₀, ω₁` | `v₃ > v₂` 是 coercivity 的前提 |
| `_relax_step` | `a, drive, … → a` | 一步松弛，含信赖域、无迹投影、球面投影 |
| `AlgebraState.decompose` | `a, τ → R, P` | `R = exp(τ skew)`、`P = exp(τ(sym−λmax I))`；`λmax` 在 float64 算并 detach |
| `AlgebraState._prefix_product` | `(B,T,k,k) → 同形` | 第 t 项是到该位置为止的左乘积；倍增步长 |
| `rho` / `rho_split` | `(a, s) → 同形` | 左乘；`rho(ab) = rho(a)rho(b)` 精确 |
| `RoPE` | `(B,heads,T,64) → 同形` | 位置旋转，保范数 |

## 可选统计量

向 `model(events, stats=st)` 传空字典即可收集。`arm='off'` 不写任何键。

| 键 | 计算方式 |
| --- | --- |
| `a_scale` | `√(mean‖a‖²_F)`，局部生成元的幅度 |
| `a_rel` | `var_t(a) / mean‖a‖²_F`，尺度无关。注意逐 token 编码下它衡量的是**位置间**差异，不是时间记忆 |
| `z_ratio` | `‖Z_T‖_F / √k`。`rot` 模式下 ≤1，不随 t 衰减（实测 T=64/256/1024 均为 0.86–0.88） |
| `z_last` | 末位置 `‖Z‖_F` |
| `R_norm_last` | 只累积正交因子时的末位置范数，构造上恒为 `√k` |
| `tau_mean`、`tau_std` | 固有时间的均值与离散度。`τ → 0` 表示结构流对该 token 完全不推进 |

`z_ratio` 的读法：接近 1 说明几乎没有收缩（近纯群），接近 0 说明乘积被压平。
后者在 `full_ap` 模式下是故障信号（实测 `z_ratio → 0.000` 时 L=64 就崩）。

这里的 `z_norm` 是 Frobenius 范数，不是谱范数。检查 `‖A‖₂ ≤ 1` 应显式用
`torch.linalg.matrix_norm(A, ord=2)`。
