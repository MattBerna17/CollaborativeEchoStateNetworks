# 🤖 Collaborative Echo State Networks

**Collaborative Echo State Networks** is a framework for predicting chaotic dynamical systems using distributed Echo State Networks (ESNs). Each ESN receives only partial (local) information about the system and collaborates with others to reconstruct the full system's behavior. The architecture supports different modes of interaction and collaboration between ESNs, with both teacher forcing (during training) and generative inference (during prediction).

---

## 🌪️ Objective

The goal is to forecast the evolution of **chaotic attractors** by modeling them through multiple interconnected ESNs, each observing only a subset of the full state space.

Given a dynamical system:
- \( \mathbf{x}(t) = [x_0(t), x_1(t), ..., x_{N-1}(t)] \)
- The ESNs aim to predict \( \mathbf{x}(t+1) \) from partial information about \( \mathbf{x}(t) \)

---

## 🧠 Training and Inference

Each ESN follows a standard Echo State Network structure:

**Training (Teacher Forcing):**
\[
h(t) = \tanh(W_{in} u(t) + W h(t-1))
\]
\[
o(t) = W_{out} h(t) \approx u(t+1)
\]

**Inference (Generative Mode):**
\[
u(t+1) = o(t)
\]

---

## 🏗️ Architectures

### 🧱 Monolithic

One ESN takes all dimensions \( x_0, ..., x_{N-1} \) as input and predicts all dimensions.

### 🔗 Chain

Each ESN \( i \) predicts \( x_{i+1}(t+1) \) from \( x_i(t) \). Sequential dependency during inference.

### ✂️ Cross

Like Chain, but one dimension is removed (e.g., \( x_j \)) and not predicted. Remaining \( N-1 \) reservoirs are used.

### 🔧 CrossStab

Extension of Cross: all reservoirs also receive \( x_j(t) \) (the removed dimension) as input to improve stability.

### 🧮 Mean

Each reservoir \( i \) takes \( x_i(t) \) and predicts all \( N \) dimensions. At inference, the predicted value for dimension \( j \) is the **mean** of the predictions from all ESNs:
\[
x_j(t+1) = \frac{1}{N} \sum_{i=0}^{N-1} x_j^{(i)}(t+1)
\]

### ⚖️ WeightedMean

Similar to Mean, but applies a **weighted average** based on training NRMSE:
\[
w_i = \frac{1 / \text{NRMSE}_i}{\sum_j 1 / \text{NRMSE}_j}
\]

Each ESN's contribution to each dimension is proportional to its accuracy during training.

---

## 🧪 Systems Supported

- **Lorenz Attractor**
- **Rössler Attractor**
- **Lorenz-96 System** with customizable dimensions

---

## 🚀 How to Run

```bash
python3 main.py --system={system} --config_file=path/to/config.json
```
Valid options for `system` are: `lorenz`, `rossler`, `lorenz96`.

(e.g. `python3 main.py --system=lorenz --config_file=configs/v1_config.json`)

---

## 📁 Structure
- `main.py`: Main entrypoint
- `esn_alternative.py`: ESN implementation
- `utils.py`: Utility functions
- `configs/`: Contains configuration files in JSON format
- `results/`: Contains results and logs