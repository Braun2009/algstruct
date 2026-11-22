# Algebraic Structure Stream

**结构是算子，靠组合计算，通过表示作用在语义上。**

一个模型同时带两条流：语义流做聚合（注意力，交换、可平均），结构流做组合
（矩阵乘法，非交换、有部分逆）。两者的接触点不是相加、不是拼接，而是代数对
向量空间的**作用**。

---

## 不是……，而是……

这些不是修辞，每一条在代码里都有对应位置。

- **不是特征，而是算子。** 结构是 $a\in\mathfrak{gl}(k)$：有乘法、有部分逆。
  向量没有乘法，所以向量不可能是结构。`model/algebra.py: AlgebraState`

- **不是加项，而是更新律本身。** 结构不在 $s+f(s)$ 的 $f$ 里，它就是状态怎么变。
  `AlgebraState.flow`

- **不进注意力，而是走组合。** 注意力对位置做加权平均：$\overline{ab}\neq\bar a\bar b$，
  分数是标量（非交换性消失），softmax 归一化（可逆性消失）。三条里任何一条都足以
  让代数退化成向量的加权和。所以结构侧只做矩阵乘与连乘扫描。
  `tests/test_representation.py: test_averaging_is_not_composition`

- **不是被存储的模式，而是状态在地形上的位置。** 没有模式库，没有路由器，没有人挑
  分支。全模型只有一组共享的生成元（8 个数），模式是 $a$ 落在哪里。
  `AlgebraState.coeffs`

- **耦合关系不是假定的物理，而是被学习的表示。** $\rho(a)s=a\cdot s$ ——左乘是
  代数作用在向量空间上的规范方式，$\rho(ab)=\rho(a)\rho(b)$ 严格成立。E/B 只是
  $\rho$ 的一个特例，不是前提。`model/action.py`

- **相互作用不是加权融合，而是同一件事的两面。** $\rho(a)Z$ 既是耦合也是一次组合。

- **隐状态不是内容缓存，而是累积变换。** $Z_t=A_tZ_{t-1}$，$Z_0=I$；$Z$ 是群元素，
  不是向量。内容只能通过改变定律（$W_dh_t\to\dot a$）影响它，不能直接加进去。

- **参数不按任务分组，而是压成一组。** 8 个生成元系数 + 两个映射，全任务共享。
  任务之间的差别来自状态，不来自参数。因此不存在"每任务未知量"，
  也就不需要为可辨识性预算做取舍。

- **有界不靠正则化，而靠极分解。** $A=RP$，$R=\mathrm{Cayley}(\mathrm{skew}\,a)$ 精确
  正交，$P=\exp(\mathrm{sym}\,a-\lVert\mathrm{sym}\,a\rVert_F I)$ 谱在 $(0,1]$。
  $\lVert A\rVert_2\le1$，故 $\lVert Z_T\rVert_2\le1$ 对任意 $T$ 成立。
  `probes/boundedness.py`

- **不可逆不靠噪声或丢弃，而靠秩亏。** $P$ 可以退化到奇异：信息可丢不可回收，
  且范数不爆。这正好是我们对"结构=不可逆运算"的要求。

- **判据不是准确率，而是不可能性。** 纯梯度流满足 $\mathrm{d}V/\mathrm{d}t\le0$，
  原理上不可能有闭轨。用一个必须有闭轨的世界去测，得到的是"能 / 不能"，
  与参数量无关。加参数永远买不到它。

- **训练脚本与探针不属于模型。** `model/` 不 import `train/` 与 `probes/`，
  出问题时可以分别定位：架构错、训练错、还是量错了。

---

## 架构

```
h (b,t,w) ──► LocalBlock ×L   (语义：聚合)
                 │
   d_t = W_d h_t                    语义 → 结构：一个源项
   ȧ   = -∇V(a) - D(a)sym a - Ω(a)skew a + d_t
   a   ← a + dt·ȧ/(1+|ȧ|/r)         编码（信任域，防显式欧拉失稳）
   A   = Cayley(skew a)·exp(sym a - ‖sym a‖ I)      ‖A‖₂ ≤ 1
   Z   ← A·Z ,  Z₀ = I               计算 = 连乘（非交换）
   h_t ← h_t + W_o·vec(Z_t)          结构 → 语义：读出
                 │
            head(norm h) → logits
```

**8 个生成元系数**（softplus 保证 $v_2,v_3,v_4\ge0$，否则 $V$ 无下界）：

$$V(a)=v_1\mathrm{Tr}\,a+v_2\mathrm{Tr}\,a^2+v_3\lVert a\rVert^2+v_4\big\lVert[a,a^\top]\big\rVert^2$$
$$D(a)=d_0+d_1\lVert a\rVert^2,\qquad\Omega(a)=\omega_0+\omega_1\lVert a\rVert^2$$

解析梯度：$\nabla_aV=v_1I+2v_2a^\top+2v_3a+4v_4[C,a]$，$C=aa^\top-a^\top a$
（与 autograd 差 $3.6\times10^{-15}$）。四项里最后一项是 Yang–Mills 型四次不变量，
非负、且仅当 $a$ 正规时为零——所以它的极小点集合是正规矩阵，能承载离散的多重极小。

**为什么 $D,\Omega$ 随状态变**：一个固定的 $\theta$ 因此能同时包含耗散区与注能区
（van der Pol 机制）。所以"全局一组参数"不等于"只有一种模式"。

---

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
tests/     15 项：表示/同态/部分逆/非交换/上界/退化谱/解析梯度/消融
docs/
  MODEL_MAP.md  架构 → 张量的逐条映射（以此为准）
```

## 明确排除

模式库、路由器、专家池、人手挑分支、把耦合展开成学出来的基、
结构经过 softmax、$\theta$ 逐 token 条件化（留作后备）、用准确率代替类型判据。

## 状态

模型搭好并通过 15 项测试；端到端冒烟通过（30 步，无死参数，上界成立）。
**尚无训练结果。** 下一步：4 臂（full / nocompose / nodrive / off）× 3 种子
× 15 任务。

## 运行

```bash
python -m probes.boundedness --horizons 10,100,3000
python -m probes.coverage --grid 5
python -m train.run --arm full --steps 1600 --device cuda
python -m unittest discover -s tests -t .
```
