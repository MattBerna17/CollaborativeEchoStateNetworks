# 🌀 Deep Echo State Networks for Lorenz Attractor Forecasting

This repository explores different architectural paradigms of **Echo State Networks (ESNs)** to model and predict the chaotic **Lorenz attractor** system. The implementations are based on modular PyTorch components and are designed for high configurability and experimentation.

---

## 📦 Project Structure

> The models are implemented using a `DeepReservoir` class composed of one or more `ReservoirModule` units. Each mode offers a different view of the input-output structure for predicting the system's future states.

---

## 🚀 Execution Modes

### 🔹 V1. **Unified Reservoir Prediction (URP)**

> A single reservoir receives all three components of the Lorenz system $[x(t), y(t), z(t)]$ as input and is trained to output the next step for all dimensions $[x(t+1), y(t+1), z(t+1)]$.

- ✅ Simple and compact
- 📉 Best suited for quick prototyping

---

### 🔹 V2. **Entangled Modular Prediction (EMP-3)**

> Three reservoirs operate independently:
- R0: input = $x(t)$ → output = $y(t+1)$
- R1: input = $y(t)$ → output = $z(t+1)$
- R2: input = $z(t)$ → output = $x(t+1)$

Each module specializes in learning a transition in a cycle, forming a predictive loop over the dimensions.

- 🔄 Cyclical dependencies
- 🧠 Good for modular interpretation

---

### 🔹 V3. **Cross-Dual Reservoirs (CDR)**

> Two reservoirs form a cross-prediction setup:
- R0: input = $x(t)$ → output = $y(t+1)$
- R1: input = $y(t)$ → output = $x(t+1)$

Dimension $z$ is completely ignored in this configuration.

- 🔍 Focused on the $x \leftrightarrow y$ subspace
- 🚫 Ignores $z(t)$, faster training

---

### 🔹 V4. **Z-Aided Cross-Dual Reservoirs (ZCDR)**

> Builds on the `CDR` architecture, but **injects ground-truth $z(t)$** as an auxiliary input:
- R0: input = $[x(t), z(t)]$ → output = $y(t+1)$
- R1: input = $[y(t), z(t)]$ → output = $x(t+1)$

This allows the network to maintain sensitivity to the full 3D state, even if only predicting two dimensions.

- 🧭 Uses $z(t)$ as a stabilizing input
- ⚠️ Can bias the network if $z(t)$ dominates

---

## ⚙️ Configuration

All modes are controlled through a JSON-based configuration file passed via:

```bash
python3 lorenz.py --config_file=config.json
```