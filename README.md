# Algebraic Structure Stream

一个序列模型，两条流。语义流做聚合：注意力、加权、可平均。结构流做组合：矩阵乘法、
非交换、有部分逆。两边只在一个地方接触，就是代数元素作用在语义向量上。

结构是 $a\in\mathfrak{gl}(k)$，$k$ 取 4 到 32。它能乘、能求逆、能不交换地复合。
向量没有这些，所以结构只能写成矩阵。

## 一步算什么

```
h      = 语义块(h)                      注意力 + MLP，因果、滑窗
d_t    = W_d h_t                        语义 → 结构，唯一入口
da     = -grad V(a) - D(a) sym a - Omega(a) skew a + d_t
a      = a + dt * da / (1 + |da|/r)     信任域，然后投影到无迹
A      = Cayley(skew a) @ exp(sym a - lmax(sym a) I)
Z      = A Z ,  Z_0 = I                 连乘，非交换
h_t   += W_o vec(Z)                     结构 → 语义
```

$A$ 的谱范数不超过 1：$\mathrm{Cayley}$ 精确正交，指数部分谱在 $(0,1]$。连乘任意多步
都不爆，不需要正则化。

7 个共享系数决定 $a$ 怎么演化：

$$V(a)=v_2\operatorname{Tr}a^2+v_3\|a\|^2+v_4\|[a,a^\top]\|^2,\qquad
D=d_0+d_1\|a\|^2,\qquad \Omega=\omega_0+\omega_1\|a\|^2$$

$D$ 和 $\Omega$ 是场，随状态变，所以一个固定的系数集能同时包含耗散区和注能区，模式
就是状态落在地形上的位置。最后一项是 Yang–Mills 型四次不变量，恒非负，仅当 $a$ 正规
时为零：它的极小点集合是全部正规矩阵，能承载多重极小。

## 包里有什么

```
model/     架构
  algebra.py   结构流：编码 / 连乘 / 读出
  action.py    作用 rho
  block.py     语义块
docs/
  DESIGN.md      为什么这么设计，假定与边界
  MODEL_MAP.md   架构到张量的逐条映射
```

训练脚本与探针不在这个包里。架构、训练、测量分开，出问题才能判断是哪一层的。

## 用

```python
from model import AlgebraConfig, AlgebraModel

cfg = AlgebraConfig(width=1536, layers=24, k=32, state_layers=(8, 16), arm='full')
model = AlgebraModel(cfg, vocab=64)          # events: (batch, time, 4) int64
logits = model(events)                       # (batch, time, vocab)
```

`arm` 四档：`full`、`nocompose`（保留每步变换，去掉累积）、
`nodrive`（切断 $W_d$，结构不再随输入变）、`off`（纯 transformer）。

## 状态

架构在跑 460M 的实验，结果还没回来。DESIGN.md 里写了假定、代价和明确不做的部分。

---

架构由 deepseek-flash 参与编写。
