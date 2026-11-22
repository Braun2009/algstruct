# 模型映射（架构 → 张量）

这一页先于代码。每个张量的形状、每个参数的数量与出处、每条判据的代码位置。
若代码与本文不一致，以本文为准并改代码。

---

## 0. 数据流

```
events (b,t,4) ──► 4×Embedding 相加 ──► h (b,t,w)
                                          │
                    ┌─────────────────────┴────────────────────┐
                    │  语义侧（交换、可平均）                  │
                    │  h ← h + LocalBlock(h)     ×L           │
                    └─────────────────────┬────────────────────┘
                                          │  在 state_layers 上插入
                                          ▼
    d_t = W_d h_t                      (b, k, k)   ← 唯一的语义入口（源项）
    ȧ_t = -∇V(a_t) - D(a_t)⊙sym(a_t) - Ω(a_t)⊙skew(a_t) + d_t
    a_t ← a_{t-1} + dt·ȧ_t              (b, k, k)   ← 编码
    A_t = Cayley(skew a_t) · P(a_t)     (b, k, k)   ← 极分解，‖A‖≤1
    Z_t = A_t Z_{t-1},   Z_0 = I        (b, k, k)   ← 计算 = 连乘（非交换）
    h_t ← h_t + W_o · vec(Z_t)          (b, t, w)   ← 读出（ρ 作用后回到残差流）
                                          │
                                   head(norm(h)) → logits
```

三问落点：编码 = 第 2–3 行；计算 = 第 5 行；耦合 = 第 1 行（语义→结构）与第 6 行（结构→语义）。

## 1. 编码：$a\in\mathfrak{gl}(k)$ 与 7 个系数

$$V(a)=v_2\mathrm{Tr}\,a^2+v_3\lVert a\rVert^2+v_4\big\lVert[a,a^\top]\big\rVert^2$$
$$D(a)=d_0+d_1\lVert a\rVert^2,\qquad \Omega(a)=\omega_0+\omega_1\lVert a\rVert^2$$

| $\theta$ | 出处 | 作用 |
|---|---|---|
| $v_2,v_3>0$ | $\mathrm{Tr}\,a^2,\ \lVert a\rVert^2$（二次） | 强制性：保证 $V$ 有下界 |
| $v_4\ge0$ | $\lVert[a,a^\top]\rVert^2$（四次，$\ge0$） | 非凸性；极小点集合 = 正规矩阵 |
| $d_0,d_1$ | 耗散场 | $d<0$ 即注能 |
| $\omega_0,\omega_1$ | 陀螺场 | 斜形，只改方向 |

为什么换掉 $\det a$：$k=3$ 时 $\det a$ 是三次，$V$ 沿 $a=tI$ 无下界（$-t^3$），
松弛没有极小点。故只用偶次不变量。

为什么没有 $v_1\mathrm{Tr}a$ 这一项：$a$ 每步被投影到无迹，$\mathrm{Tr}a\equiv0$，
这一项的梯度恒为零，是个死参数。系数因此是 7 个，不是 8 个。

解析梯度（已验证与 autograd 差 $3.6\times10^{-15}$）：

$$\nabla_a V=2v_2a^\top+2v_3a+4v_4[C,a],\qquad C=aa^\top-a^\top a$$

$v_2,v_3,v_4$ 用 softplus 保证为正。

## 2. 计算：连乘为什么有界

$$A=RP,\quad R=\mathrm{Cayley}\big(\tfrac{a-a^\top}{2}\big),\quad P=\exp\Big(\tfrac{a+a^\top}{2}-\lambda_{\max}\big(\tfrac{a+a^\top}{2}\big)I\Big)$$

$\lVert R\rVert=1$（精确正交）、$\lVert P\rVert\le1$（谱在 $(0,1]$）⇒ $\lVert A\rVert\le1$ ⇒
$\lVert Z_T\rVert=\lVert A_T\cdots A_1\rVert\le1$。不需要正则化。

$P$ 可以退化到秩亏 ⇒ 部分逆失效，但范数不爆。

## 3. 耦合：一处进、一处出

| 方向 | 载体 | 性质 | 判据 |
|---|---|---|---|
| 语义 → 结构 | $d_t=W_dh_t$ 加进 $\dot a$ | 改定律（源），不改状态 | 切断 $W_d$ ⇒ $a_t$ 常值、$Z$ 退化成 $A^t$ |
| 结构 → 语义 | $h\leftarrow h+W_o\mathrm{vec}(Z_t)$ | $\rho$ 左乘后再读出 | $\rho(AB)=\rho(A)\rho(B)$；结构不经过 softmax |

不往 $Z$ 里注入。内容只经 $d_t\to a_t\to A_t$ 影响 $Z$。

## 4. 参数表

| 项 | 形状 | 数量（$k=3,w=192$） | 共享范围 |
|---|---|---|---|
| $\theta$ | 7 | 7 | 全局（全任务、全 token） |
| $W_d$ | $w\to k^2$ | 1737 | 每 state layer |
| $W_o$ | $k^2\to w$ | 1737 | 每 state layer |
| 语义主干 | — | 其余 | 通常 |

没有每任务参数，因此不触发"未知量 ≤ 观测/4"。

## 5. 判据与代码位置

| 判据 | 含义 | 在哪 |
|---|---|---|
| $\lVert A\rVert\le1$ | 连乘有界 | `decompose()` + 训练时记录 `mat_norm` |
| $\mathrm{Var}_t(a_t)>0$ | 结构流没死（不是单一固定变换） | 训练时记录 `a_var` |
| $\lVert Z_T\rVert$ 不指数衰减 | 记忆没有塌掉 | 记录 `z_norm` |
| $\rho$ 同态 | 耦合是表示 | `action.py: rho`（测试不在本仓库） |
| 覆盖率 | 固定 $\theta$，$a$ 扫过时能出现多少种局部类型 | 待写 `tools/coverage.py` |
| 任务可分性 | 任务在有效场空间是否分开 | 待写 |

## 6. 排除

模式库、路由器、人手挑分支、$\pi$ 三比特、$\theta$ 逐 token 条件化（留作后备）、
结构经过 softmax、把结构展开成学出来的基。
