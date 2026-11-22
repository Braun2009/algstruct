# Algebraic Structure Stream

模型里跑两条流。

语义流做聚合：注意力按位置加权平均，交换、可平均，可逆性被 softmax 归一化吃掉。
结构流做组合：矩阵乘法，非交换，只有部分逆。

两条流在代数对向量空间的左乘作用上接触，$\rho(a)s=a\cdot s$，
$\rho(ab)=\rho(a)\rho(b)$ 严格成立。耦合是这一次乘法，组合也是这一次乘法。

## 结构流

状态 $a\in\mathfrak{gl}(k)$ 是算子。向量没有乘法，所以结构不能存在向量里。
$a$ 落在 $\mathfrak{gl}(k)$ 的哪个位置，就决定模型处在哪种模式。没有模式库，没有
路由器，没有人挑分支：全模型只有一组共享的生成元（8 个数）。
（`model/algebra.py: AlgebraState.coeffs`）

$a$ 的更新律

$$\dot a = -\nabla V(a) - D(a)\,\mathrm{sym}\,a - \Omega(a)\,\mathrm{skew}\,a + d_t$$

$d_t = W_d h_t$ 是语义侧来的源项，内容只能从这里进入结构流，作用方式是扰动定律。
显式欧拉会失稳，所以每步做一次信任域收缩：

$$a \leftarrow a + \mathrm{d}t\,\dot a \Big/ \big(1 + |\dot a|/r\big)$$

（`AlgebraState.flow`）

$D$ 和 $\Omega$ 随状态变，$D=d_0+d_1\lVert a\rVert^2$，
$\Omega=\omega_0+\omega_1\lVert a\rVert^2$。固定的一组 $\theta$ 因此可以同时包含
耗散区和注能区（van der Pol 机制），全局一组参数不等于只有一种行为。

**8 个生成元系数** $v_{1..4},d_0,d_1,\omega_0,\omega_1$，其中 $v_2,v_3,v_4$ 过 softplus
保证非负，否则 $V$ 无下界：

$$V(a)=v_1\mathrm{Tr}\,a+v_2\mathrm{Tr}\,a^2+v_3\lVert a\rVert^2+v_4\big\lVert[a,a^\top]\big\rVert^2$$

解析梯度 $\nabla_aV=v_1I+2v_2a^\top+2v_3a+4v_4[C,a]$，$C=aa^\top-a^\top a$，与 autograd
差 $3.6\times10^{-15}$。最后一项是 Yang–Mills 型四次不变量，非负，且仅当 $a$ 正规时
取零。所以极小点集合是正规矩阵，能承载离散的多重极小。

## 计算

$a$ 先经极分解映射成 $A$：

$$A=\mathrm{Cayley}(\mathrm{skew}\,a)\cdot\exp\big(\mathrm{sym}\,a-\lVert \mathrm{sym}\,a\rVert_F I\big)$$

左边精确正交，右边谱落在 $(0,1]$，所以 $\lVert A\rVert_2\le1$，进而
$\lVert Z_T\rVert_2\le1$ 对任意 horizon 成立，不需要任何正则化
（`probes/boundedness.py`）。右边可以退化到奇异：信息能丢，丢掉的收不回来，范数
也不爆。结构作为不可逆运算的要求由秩亏满足，不靠噪声或丢弃。

计算是连乘，$Z_t=A_tZ_{t-1}$，$Z_0=I$。$Z$ 是群元素，不能直接往里加内容。
语义侧取 $h_t\leftarrow h_t+W_o\,\mathrm{vec}(Z_t)$，最后 `head(norm h)` 出 logits。

```
h (b,t,w) ──► LocalBlock ×L        语义：聚合
                 │
   d_t = W_d h_t                   语义 → 结构：源项
   ȧ   = -∇V(a) - D(a)sym a - Ω(a)skew a + d_t
   a   ← a + dt·ȧ/(1+|ȧ|/r)        编码：信任域
   A   = Cayley(skew a)·exp(sym a - ‖sym a‖ I)     ‖A‖₂ ≤ 1
   Z   ← A·Z ,  Z₀ = I             计算：连乘（非交换）
   h_t ← h_t + W_o·vec(Z_t)        结构 → 语义：读出
                 │
            head(norm h) → logits
```

## 判据

纯梯度流满足 $\mathrm{d}V/\mathrm{d}t\le0$，闭轨在这种动力学里不可能出现。拿一个必须
有闭轨的世界去测，输出只有「能 / 不能」两种，与参数量无关，加参数也买不到
（`tests/test_representation.py`）。用准确率当判据会把这个结论盖掉，所以测试量的是
不可能性。

## 目录

```
model/     架构。不 import 训练与探针
  algebra.py   结构流：编码 / 计算 / 读出
  action.py    作用 ρ
  block.py     语义块（局部因果注意力 + MLP）
train/     训练。只 import model
  run.py       15 任务联合训练与逐任务评测
  bench.py     任务生成器（基准，不属于架构）
probes/    探针。只 import model，用来量架构性质，不做训练
  boundedness.py  ‖A‖、‖Z‖ 随 horizon 的行为
  coverage.py     固定 θ，扫过 gl(k)，数一个定律能产生几种局部类型
tests/     15 项：表示 / 同态 / 部分逆 / 非交换 / 上界 / 退化谱 / 解析梯度 / 消融
docs/
  MODEL_MAP.md  架构 → 张量的逐条映射（以此为准）
```

`model/` 不 import `train/` 与 `probes/`，出问题时可以分开定位：架构错、训练错、
还是量错了。

## 没做的

模式库、路由器、专家池、人手挑分支、把耦合展开成学出来的基、结构经过 softmax、
$\theta$ 逐 token 条件化（留作后备）、用准确率代替类型判据。

## 状态

模型搭好，15 项测试通过，端到端冒烟通过（30 步，无死参数，上界成立）。
还没有训练结果。下一步是 4 臂（full / nocompose / nodrive / off）× 3 种子 × 15 任务。

## 运行

```bash
python -m probes.boundedness --horizons 10,100,3000
python -m probes.coverage --grid 5
python -m train.run --arm full --steps 1600 --device cuda
python -m unittest discover -s tests -t .
```

## 许可

[Apache License 2.0](LICENSE)，Copyright 2026 Braun2009。

可自由使用、修改、分发，含商业用途。条件：保留版权声明与署名、标注改动、随附一份
许可证副本。许可证带显式专利授权；就本项目提起专利诉讼时，授权终止。
