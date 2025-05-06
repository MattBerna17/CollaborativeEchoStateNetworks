# 🌪️ Echo State Networks for Chaotic System Forecasting

This project explores **modular Echo State Networks (ESNs)** to forecast multiple chaotic systems with varying prediction topologies and reservoir configurations. It is designed to be flexible, scalable, and well-suited for comparative analysis across different coupling strategies and datasets.

---

## 🧠 Supported Dynamical Chaotic Systems

You can simulate and forecast three well-known chaotic systems:

- **lorenz** → 3D Lorenz attractor
- **rossler** → 3D Rössler system
- **lorenz96** → 4D Lorenz96 model

---

## 🔮 Prediction Architectures

Each system can be used with the following four architectures:

- **UNIMODE** 🧱  
  A single reservoir receives and predicts all dimensions together.

- **CHAIN** 🔗  
  A modular chain of reservoirs where each reservoir $i$ receives dimension $i$ and predicts dimension $i+1$.

- **CROSS** 🔁  
  Two or more reservoirs, each receiving different input dimensions and predicting complementary outputs.

- **CROSS+STAB** 🧭  
  Same as **CROSS**, but each reservoir also receives one stabilizing ground truth dimension (e.g., $z$ or $x_3$) as additional input.

These names generalize across systems with different dimensionalities.

---

## 🚀 How to Run

Use the following command to launch a simulation:

```
python3 main.py --system=s --config_file=path/to/config.json**
```

Replace:

- **s** with one of: `lorenz`, `rossler`, or `lorenz96`
- **path/to/config.json** with the actual path to your configuration file

Example:

```bash
python3 main.py --system=lorenz --config_file=configs/lorenz_config.json
```
---

## 📁 Configuration

The model behavior and reservoir parameters are entirely driven by JSON config files. You can define:

- Number of reservoirs and their connections
- Input/output dimensions per module
- Stabilizing signals
- Activation units, spectral radius, regularization, etc.

See example config files in the `configs/` directory.

---

## 📊 Visualization

Plots for prediction vs target can be displayed depending on config settings.

---

## 🧪 Citation & Credits

Gallicchio,  C.,  Micheli,  A.,  Pedrelli,  L.: Deep  reservoir  computing:
A  critical  experimental  analysis.    Neurocomputing268,  87–99  (2017).
https://doi.org/10.1016/j.neucom.2016.12.08924.

Developed by Matthew Bernardi, a Computer Science student, as part of a bachelor thesis project on modular ESNs and robust forecasting.

---