# 🌐 Network Resilience Analysis

**Scale-Free Network Resilience Analysis: Hub Attack Simulation and GAT-Based Node Criticality Prediction**

An academic-grade research project that models landscape connectivity as a scale-free network and simulates the impact of targeted habitat destruction (hub attacks) on network robustness. This project also employs Graph Attention Networks (GATs) for predicting ecological node criticality and generates interactive 3D-like physics-based visualizations to demonstrate ecosystem collapse.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Methodology & Stages](#-methodology--stages)
    - [Stage 1: Network Generation](#stage-1-network-generation)
    - [Stage 2: Hub Attack Simulation](#stage-2-hub-attack-simulation)
    - [Stage 3: GAT-Based Criticality Prediction](#stage-3-gat-based-criticality-prediction)
    - [Stage 4: Interactive HTML Visualizations](#stage-4-interactive-html-visualizations)
3. [Installation & Setup](#-installation--setup)
4. [Usage](#-usage)
5. [Key Scientific Findings](#-key-scientific-findings)
6. [Results & Visualizations](#-results--visualizations)
7. [References](#-references)

---

## 📋 Overview

This project uses the **Barabási-Albert model** to generate scale-free networks representing landscape connectivity. In conservation biology, landscape corridors and core patches form networks where certain nodes act as critical hubs. We analyze network resilience under target attacks vs. random degradation, train a custom **Graph Attention Network (GAT)** to predict a multi-factor structural criticality score (Impact Score), and visualize the resulting fragmentation using **Pyvis**.

---

## 🧪 Methodology & Stages

### Stage 1: Network Generation
- Generates a Scale-Free network ($N=2000$, $m=3$) using the Barabási-Albert model.
- Calculates and logs baseline topological metrics:
  - Total Nodes / Edges
  - Giant Component (GC) Size Ratio (percentage)
  - Average Node Degree
- Plots the Node Degree Distribution as a log-scale histogram to verify the power-law characteristics.

### Stage 2: Hub Attack Simulation
- Evaluates the structural robustness of the ecosystem under three targeted sequential node removal strategies:
  1. **Degree Centrality Attack**: Targeting highest degree nodes first.
  2. **Betweenness Centrality Attack**: Targeting nodes that lie on the most shortest paths.
  3. **Random Removal**: Baseline simulation representing non-targeted, stochastic habitat loss.
- Plots the network collapse curves tracking the decline of the Giant Component Size Ratio.

### Stage 3: GAT-Based Criticality Prediction
- **Data Prep (Label Generation)**: Generates a continuous "Impact Score" for each node based on the structural damage caused by its removal (number of new components created, edges lost, and decrease in GC size).
- **Feature Extraction**: Computes normalized local features for each node: Degree, Clustering Coefficient, Betweenness Centrality, and PageRank.
- **Model Architecture**: Trains a 3-layer Graph Attention Network (GAT) with multi-head attention to regress node criticality scores.
- **Evaluation**: Simulates a network attack targeting nodes in order of the GAT-predicted criticality and compares the collapse curve against classical metrics.

### Stage 4: Interactive HTML Visualizations
- Builds an interactive visualization of the network (scaled down to $N=300$ for smooth web browser rendering) using `pyvis`.
- Colors the top 5% highest-degree hubs in a warning color (red/orange) and scales node sizes based on their degree.
- Generates two files:
  1. `interactive_ecosystem_demo.html`: The fully intact scale-free ecosystem network with custom legends and hover tooltips.
  2. `interactive_ecosystem_attacked.html`: The exact same network with the top 5% hubs removed, illustrating structural fragmentation.

---

## 🚀 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/isglyv/network-resilience-analysis.git
   cd network-resilience-analysis
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

Execute the pipeline step-by-step:

```bash
# Stage 1: Generate network and plot degree distribution
python stage1_network_generation.py

# Stage 2: Run hub attack simulation and compare classical metrics
python stage2_attack_simulation.py

# Stage 3: Train GAT model and compare GAT predictions in attack simulation
python stage3_gnn_training.py

# Stage 4: Generate interactive HTML visualizations (Intact vs Attacked)
python stage4_interactive_demo.py
```

---

## 🔬 Key Scientific Findings

1. **Scale-Free Fragility**: The BA network remains highly robust against random node removal, retaining its giant component even after 50% of the nodes are removed.
2. **Hub Vulnerability**: Target attacks on hubs (highest degree or betweenness centrality) lead to rapid network fragmentation, causing a complete collapse of connectivity when less than 10-15% of hubs are removed.
3. **GAT Predictive Performance**: The GAT model accurately predicts composite node criticality (Test MSE: **~0.0017**) and can successfully guide attack strategies that are almost as lethal as perfect, computationally expensive degree attacks.

---

## 📊 Results & Visualizations

| Visualization | Description |
|---|---|
| `degree_distribution.png` | Log-scale degree distribution confirming the scale-free nature. |
| `network_collapse_simulation.png` | Network collapse curves under Degree, Betweenness, and Random attacks. |
| `stage3_gat_collapse_comparison.png` | Comparison curve including the GAT-predicted attack sequence. |
| `interactive_ecosystem_demo.html` | Interactive intact ecosystem visualization with highlighted hubs. |
| `interactive_ecosystem_attacked.html` | Interactive fragmented ecosystem visualization showing the post-attack state. |

---

## 📖 References

- Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509-512.
- Albert, R., Jeong, H., & Barabási, A.-L. (2000). Error and attack tolerance of complex networks. *Nature*, 406(6794), 378-382.
- Veličković, P., et al. (2018). Graph Attention Networks. *ICLR*.

---

## 📄 License

MIT License
