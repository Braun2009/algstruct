# v5_tau — 结构固有时间

## 改动

每个 token 有一个正的固有时间步长，ODE 与李群积分都写在 τ 上，而不是 token 编号：

$$\Delta\tau_t = \mathrm{dt}\cdot\sigma(w_\tau\cdot\mathrm{LN}(h_t)) \in (0, \mathrm{dt})$$

$$\frac{da}{d\tau} = -\nabla V(a) - D(a)\,\mathrm{sym}(a) - \Omega(a)\,\mathrm{skew}(a) + d_t,
\qquad \frac{dZ}{d\tau} = \mathrm{skew}(a)\,Z$$

## 它说明了什么

真实序列里改变结构的事件只占少数。`group_state` 的 128 个位置中约 7 个有效，
其余是填充。固定步长下每个填充位置都会推进一次 ODE 并乘一次矩阵，序列长度因此进入结果；
引入 τ 后，`τ → 0` 的 token 不推进结构时钟。

本变体只把 τ 乘到 `R` 上，`P` 未乘，因此 `τ → 0` 时乘积仍在被收缩。
`v6_hom` 让两个因子都乘 τ，使 `A_t = I` 严格成立。

## 结果

`s3_extrap`、`L_train = 16`：`nocompose` 在训练长度上达到 0.995，L=32 降到 0.236。
没有连乘却记住了 16 个 token，说明记忆走的是 `a_t` 的跨时间递推。
