# 🌐 Network Resilience Analysis

**Scale-Free Network Resilience Analysis: Hub Attack Simulation for Landscape Connectivity**

An academic-grade Python project that models landscape connectivity as a scale-free network and simulates the impact of targeted habitat destruction (hub attacks) on network robustness.

## 📋 Overview

This project uses the **Barabási-Albert model** to generate a scale-free network representing landscape connectivity. It then simulates targeted attacks on hub nodes to analyze network resilience — a critical concept in conservation biology and landscape ecology.

## 🧪 Methodology

### Stage 1: Network Generation
- Generates a Scale-Free network (N=2000, m=3) using the Barabási-Albert model
- Calculates baseline metrics: nodes, edges, giant component ratio, average degree
- Visualizes the power-law degree distribution

### Stage 2: Hub Attack Simulation
- Computes Degree Centrality and Betweenness Centrality for all nodes
- Simulates three attack strategies:
  - **Degree Centrality Attack** — removes highest-degree nodes first
  - **Betweenness Centrality Attack** — removes highest-betweenness nodes first
  - **Random Removal** — baseline comparison
- Plots network collapse curves showing giant component degradation

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run Stage 1: Network Generation
python stage1_network_generation.py

# Run Stage 2: Attack Simulation
python stage2_attack_simulation.py
```

## 📦 Dependencies

- Python 3.8+
- NetworkX
- Matplotlib
- NumPy

## 📊 Output

| File | Description |
|------|-------------|
| `degree_distribution.png` | Log-scale histogram of node degree distribution |
| `network_collapse_simulation.png` | Comparative collapse curves under different attack strategies |

## 📖 References

- Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509-512.
- Albert, R., Jeong, H., & Barabási, A.-L. (2000). Error and attack tolerance of complex networks. *Nature*, 406(6794), 378-382.

## 📄 License

MIT License
