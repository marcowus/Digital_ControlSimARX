# 高级预设时间自适应控制系统原理说明

本项目演示了一种基于**预设时间控制 (Prescribed-Time Control)**、**Koopman算子 (Koopman Operator)** 和 **自适应死区逆 (Adaptive Dead-zone Inverse)** 的高级控制算法，用于解决具有未知非线性动力学和执行器死区的三阶严格反馈系统的轨迹跟踪问题。

## 1. 系统模型 (System Model)

被控对象为一个三阶严格反馈非线性系统：

$$
\begin{cases}
\dot{x}_1 = x_2 + f_1(x_1) + d_1(t) \\
\dot{x}_2 = x_3 + f_2(x_2) + d_2(t) \\
\dot{x}_3 = u(v)
\end{cases}
$$

其中：
- $x = [x_1, x_2, x_3]^T$ 为系统状态。
- $f_1(x_1) = \sin(x_1^2)\cos(x_1)$, $f_2(x_2) = \sin(x_2)(x_2 + x_2^2)$ 为未知的非线性漂移函数。
- $d_1(t), d_2(t)$ 为外部扰动。
- $u(v)$ 为执行器输出，受死区（Dead-zone）非线性影响，$v$ 为控制指令。

### 1.1 执行器死区模型 (Actuator Dead-zone)

执行器存在非对称死区特性：

$$
u(v) = \begin{cases}
m_r v + b_r & \text{if } v \ge 0.75 \\
m_l v + b_l & \text{if } v < -0.5 \\
0 & \text{otherwise}
\end{cases}
$$

在代码仿真中，设定参数为 $m_r=1.2, b_r=-0.9$ 和 $m_l=0.8, b_l=0.4$。
控制器的目标是通过自适应律估计死区参数，并计算补偿后的指令 $v$。

## 2. 预设时间缩放 (Strict Prescribed-Time Scaling)

为了在预设时间 $T$ 内实现收敛，引入缩放函数 $\mu(t)$：

$$
\mu(t) = \left( \frac{T}{T - t} \right)^p
$$

当 $t \to T$ 时，$\mu(t) \to \infty$。
通过构建变换后的状态 $\xi_i = \mu(t) z_i$（其中 $z_i$ 为跟踪误差），并迫使 $\xi_i$ 有界，可以保证原误差 $z_i$ 以 $1/\mu(t)$ 的速度衰减，从而在 $T$ 时刻收敛至零。

## 3. Koopman 算子数据驱动估计 (Koopman Operator)

为了处理未知的非线性函数 $f_i(x_i)$，系统采用 Koopman 算子理论进行线性化估计。
Koopman 算子将非线性系统的状态空间提升到无限维（或高维）的观测函数空间，使得非线性演化变为线性演化。

定义提升函数（Observables）$\Psi(x)$，包括多项式项、三角函数项以及ReLU项（用于捕捉死区特征）：

$$
\Psi(x) = [x_1, x_2, x_3, x_1^2, ..., \sin(x_1), ..., \max(0, x_1), ...]^T
$$

通过收集轨迹数据 $(X, Y, U)$，利用最小二乘法求解近似的 Koopman 矩阵 $A_{aug}$：

$$
\min_{A_{aug}} \sum \| \Psi(x_{k+1}) - A_{aug} [\Psi(x_k)^T, u_k]^T \|^2
$$

在控制器中，利用 $A_{aug}$ 预测系统的漂移项 $f(x)$，作为前馈补偿。

## 4. 控制器设计 (Command Filtered Control)

采用指令滤波反步法 (Command Filtered Backstepping) 设计控制器，避免了传统反步法中的“计算膨胀”问题。

### 4.1 误差定义
- 跟踪误差：$z_1 = x_1 - y_d$
- 虚拟控制误差：$z_2 = x_2 - \alpha_1$, $z_3 = x_3 - \alpha_2$

### 4.2 虚拟控制律
对于每一步 $i=1, 2$，设计虚拟控制量 $\alpha_i$：

$$
\alpha_{i, raw} = \frac{-k_i \xi_i - \text{bar}(\xi_i)}{\mu} - \frac{\dot{\mu}}{\mu} z_i - \hat{f}_i + \dot{\alpha}_{i-1}
$$

其中 $\hat{f}_i$ 来自 Koopman 预测，$\text{bar}(\xi_i)$ 为障碍李雅普诺夫函数（Barrier Lyapunov Function）项，用于约束状态。

通过二阶指令滤波器 (Command Filter) 获得平滑后的 $\alpha_i$ 及其导数 $\dot{\alpha}_i$：

$$
\ddot{\alpha}_i = -2\zeta\omega_n \dot{\alpha}_i - \omega_n^2 (\alpha_i - \alpha_{i, raw})
$$

### 4.3 自适应死区逆 (Adaptive Inverse)

最终控制律 $u_{des}$ 计算得到后，通过自适应逆模型计算实际输入 $v$：

$$
v = \frac{u_{des} - \hat{b}}{\hat{m}}
$$

自适应律基于李雅普诺夫分析设计（简化形式）：

$$
\dot{\hat{m}} = \gamma \cdot \xi_3 \cdot v
$$
$$
\dot{\hat{b}} = \gamma \cdot \xi_3
$$

并配合投影算子 (Projection Operator) 防止参数漂移越界。

## 5. 总结

该算法结合了模型驱动（反步法、预设时间缩放）与数据驱动（Koopman 估计）的优势，能够在系统模型未知且存在执行器死区的情况下，实现高精度的预设时间轨迹跟踪。
