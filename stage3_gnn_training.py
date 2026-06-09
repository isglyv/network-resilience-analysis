"""
Stage 3: Graph Attention Network (GAT) for Ecological Node Criticality
========================================================================
Trains a GAT model using PyTorch Geometric to predict node criticality
(impact score) and compares its attack simulation performance against
classical centrality-based strategies.

Pipeline:
    1. Label Generation — compute true impact score per node
    2. Feature Extraction — degree + clustering coefficient
    3. PyTorch Geometric data conversion with train/test split
    4. GAT model training (regression)
    5. GAT-based attack simulation
    6. Comparative visualization with all 4 strategies

Author: Network Resilience Analysis Project
Date: 2026
"""

import sys
import copy
import random
import logging
import warnings

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx

from stage1_network_generation import generate_scale_free_network, calculate_baseline_metrics
from stage2_attack_simulation import (
    compute_centrality_metrics,
    simulate_attack,
    simulate_random_removal,
    get_giant_component_size,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration & Constants
# =============================================================================

RANDOM_SEED = 42
TRAIN_RATIO = 0.80          # 80/20 train/test split
NUM_EPOCHS = 200            # Training epochs
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
HIDDEN_CHANNELS = 64        # Hidden layer size
NUM_HEADS = 4               # Number of attention heads
DROPOUT = 0.3

PLOT_DPI = 300
PLOT_FIGSIZE = (13, 8)
COLLAPSE_PLOT_FILE = "stage3_gat_collapse_comparison.png"

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Step 1: Label Generation — Impact Score Calculation
# =============================================================================

def compute_impact_scores(G: nx.Graph) -> dict:
    """
    Compute the true Impact Score for each node.

    Uses a composite criticality metric that captures the real structural
    damage caused by removing a single node:
        - Number of connected components created after removal
        - Number of edges lost (degree of the removed node)
        - Decrease in Giant Component size

    The final score is a weighted combination normalized to produce
    a diverse, continuous distribution suitable for regression.

    Parameters
    ----------
    G : nx.Graph
        The input network graph.

    Returns
    -------
    dict
        Dictionary mapping node -> impact score (float).
    """
    logger.info("Computing composite impact scores for all nodes...")

    original_gc_size = get_giant_component_size(G)
    original_num_components = nx.number_connected_components(G)
    impact_scores = {}

    nodes = list(G.nodes())
    total = len(nodes)

    for i, node in enumerate(nodes):
        G_temp = G.copy()
        degree = G_temp.degree(node)
        G_temp.remove_node(node)

        if G_temp.number_of_nodes() > 0:
            new_gc_size = get_giant_component_size(G_temp)
            new_num_components = nx.number_connected_components(G_temp)
        else:
            new_gc_size = 0
            new_num_components = 0

        # Component 1: Fragmentation — how many new components are created
        fragmentation = new_num_components - original_num_components

        # Component 2: Edge loss — degree of removed node (connectivity damage)
        edge_loss = degree

        # Component 3: GC size decrease
        gc_decrease = original_gc_size - new_gc_size

        # Composite score: weighted sum
        # Fragmentation is heavily weighted as it indicates structural criticality
        impact = (fragmentation * 10.0) + (edge_loss * 1.0) + (gc_decrease * 5.0)
        impact_scores[node] = impact

        # Progress logging every 20%
        if (i + 1) % (total // 5) == 0:
            logger.info(f"  Impact score progress: {i + 1}/{total} nodes "
                        f"({(i + 1) / total * 100:.0f}%)")

    scores = list(impact_scores.values())
    logger.info(f"Impact scores computed. "
                f"Min: {min(scores):.2f}, Max: {max(scores):.2f}, "
                f"Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")

    return impact_scores


# =============================================================================
# Step 2: Feature Extraction
# =============================================================================

def extract_node_features(G: nx.Graph) -> np.ndarray:
    """
    Extract and normalize local node features.

    Features per node (4 features):
        - Degree (normalized)
        - Clustering Coefficient
        - Betweenness Centrality
        - PageRank

    Parameters
    ----------
    G : nx.Graph
        The input network graph.

    Returns
    -------
    np.ndarray
        Node feature matrix of shape (num_nodes, 4).
    """
    logger.info("Extracting node features (degree, clustering, betweenness, pagerank)...")

    nodes = sorted(G.nodes())

    # Feature 1: Degree
    degrees = np.array([G.degree(n) for n in nodes], dtype=np.float32)

    # Feature 2: Clustering Coefficient
    clustering = nx.clustering(G)
    cc = np.array([clustering[n] for n in nodes], dtype=np.float32)

    # Feature 3: Betweenness Centrality
    betweenness = nx.betweenness_centrality(G)
    bc = np.array([betweenness[n] for n in nodes], dtype=np.float32)

    # Feature 4: PageRank
    pagerank = nx.pagerank(G)
    pr = np.array([pagerank[n] for n in nodes], dtype=np.float32)

    # Normalize all features to [0, 1] using min-max scaling
    def minmax_normalize(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)

    degrees_norm = minmax_normalize(degrees)
    cc_norm = minmax_normalize(cc)
    bc_norm = minmax_normalize(bc)
    pr_norm = minmax_normalize(pr)

    # Stack into feature matrix [num_nodes, 4]
    X = np.column_stack([degrees_norm, cc_norm, bc_norm, pr_norm])

    logger.info(f"Feature matrix shape: {X.shape}")
    return X


# =============================================================================
# Step 3: PyTorch Geometric Data Object
# =============================================================================

def create_pyg_data(
    G: nx.Graph,
    features: np.ndarray,
    impact_scores: dict,
    train_ratio: float = TRAIN_RATIO,
) -> Data:
    """
    Convert NetworkX graph, features, and labels into a PyG Data object.

    Creates boolean train/test masks with the specified split ratio.

    Parameters
    ----------
    G : nx.Graph
        The input network graph.
    features : np.ndarray
        Node feature matrix of shape (num_nodes, num_features).
    impact_scores : dict
        Dictionary mapping node -> impact score.
    train_ratio : float
        Fraction of nodes to use for training.

    Returns
    -------
    torch_geometric.data.Data
        The PyG data object with x, edge_index, y, train_mask, test_mask.
    """
    logger.info("Converting to PyTorch Geometric Data object...")

    nodes = sorted(G.nodes())
    num_nodes = len(nodes)

    # Node features
    x = torch.tensor(features, dtype=torch.float)

    # Labels (impact scores) — sorted by node index
    y = torch.tensor(
        [impact_scores[n] for n in nodes], dtype=torch.float
    ).unsqueeze(1)  # Shape: [num_nodes, 1]

    # Normalize labels to [0, 1] for stable training
    y_min, y_max = y.min(), y.max()
    if y_max > y_min:
        y_normalized = (y - y_min) / (y_max - y_min)
    else:
        y_normalized = torch.zeros_like(y)

    # Edge index from NetworkX
    # Create mapping from node to contiguous index
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = list(G.edges())
    edge_list = []
    for u, v in edges:
        edge_list.append([node_to_idx[u], node_to_idx[v]])
        edge_list.append([node_to_idx[v], node_to_idx[u]])  # Undirected

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Train/Test split (random)
    perm = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:train_size]] = True
    test_mask[perm[train_size:]] = True

    # Build Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_normalized,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    # Store original (unnormalized) labels and normalization params for later
    data.y_original = y
    data.y_min = y_min
    data.y_max = y_max
    data.node_list = nodes

    logger.info(f"Data object: {data}")
    logger.info(f"  Train nodes: {train_mask.sum().item()}, "
                f"Test nodes: {test_mask.sum().item()}")

    return data


# =============================================================================
# Step 4: GAT Model Architecture
# =============================================================================

class GATRegressor(torch.nn.Module):
    """
    Graph Attention Network for node-level regression.

    Architecture:
        - GATConv Layer 1: input_dim → hidden_dim (multi-head)
        - GATConv Layer 2: hidden_dim * heads → hidden_dim (multi-head)
        - GATConv Layer 3: hidden_dim * heads → 1 (single output, 1 head)

    The model predicts a single continuous value per node
    (the ecological impact score).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = HIDDEN_CHANNELS,
        heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # Layer 1: Multi-head attention
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, dropout=dropout
        )

        # Layer 2: Multi-head attention
        self.conv2 = GATConv(
            hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout
        )

        # Layer 3: Single-head output (regression)
        self.conv3 = GATConv(
            hidden_channels * heads, 1, heads=1, concat=False, dropout=dropout
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        """Forward pass through the 3-layer GAT."""
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3 (output)
        x = self.conv3(x, edge_index)

        return x


# =============================================================================
# Step 5: Training Pipeline
# =============================================================================

def train_gat_model(data: Data, num_epochs: int = NUM_EPOCHS) -> GATRegressor:
    """
    Train the GAT model to predict node impact scores.

    Parameters
    ----------
    data : Data
        PyG Data object with features, labels, and masks.
    num_epochs : int
        Number of training epochs.

    Returns
    -------
    GATRegressor
        The trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    model = GATRegressor(in_channels=data.x.size(1)).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.MSELoss()

    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Training for {num_epochs} epochs...")

    best_test_mse = float("inf")

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])

        train_loss.backward()
        optimizer.step()

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index)
            test_loss = criterion(pred[data.test_mask], data.y[data.test_mask])

        if test_loss.item() < best_test_mse:
            best_test_mse = test_loss.item()

        # Log progress
        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:>4d}/{num_epochs} | "
                f"Train MSE: {train_loss.item():.6f} | "
                f"Test MSE: {test_loss.item():.6f}"
            )

    print("\n" + "=" * 55)
    print("  GAT TRAINING RESULTS")
    print("=" * 55)
    print(f"  Final Train MSE:  {train_loss.item():.6f}")
    print(f"  Final Test MSE:   {test_loss.item():.6f}")
    print(f"  Best Test MSE:    {best_test_mse:.6f}")
    print("=" * 55 + "\n")

    return model


# =============================================================================
# Step 6: GAT-Based Attack Simulation
# =============================================================================

def get_gat_predictions(model: GATRegressor, data: Data) -> dict:
    """
    Get GAT predicted impact scores for all nodes.

    Parameters
    ----------
    model : GATRegressor
        Trained GAT model.
    data : Data
        PyG Data object.

    Returns
    -------
    dict
        Dictionary mapping original node ID -> predicted impact score.
    """
    device = next(model.parameters()).device
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).squeeze().cpu().numpy()

    # Map predictions back to original node IDs
    nodes = data.node_list
    pred_dict = {nodes[i]: float(predictions[i]) for i in range(len(nodes))}

    logger.info(f"GAT predictions obtained for {len(pred_dict)} nodes.")
    return pred_dict


# =============================================================================
# Step 7: Final Visualization — 4-Strategy Collapse Comparison
# =============================================================================

def plot_four_strategy_collapse(
    results: dict,
    initial_nodes: int,
    output_file: str = COLLAPSE_PLOT_FILE,
    figsize: tuple = PLOT_FIGSIZE,
    dpi: int = PLOT_DPI,
) -> None:
    """
    Plot the Network Collapse Curve with 4 strategies.

    Includes: Degree Attack, Betweenness Attack, Random Removal,
    and GAT Predicted Attack (prominent thick line).

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy name -> (fractions_removed, gc_sizes).
    initial_nodes : int
        Total number of nodes in the original network.
    output_file : str
        Filename for the saved plot.
    figsize : tuple
        Figure dimensions.
    dpi : int
        Plot resolution.
    """
    style_config = {
        "Degree Centrality Attack": {
            "color": "#FF7043",
            "linestyle": "--",
            "linewidth": 2.0,
            "marker": "o",
            "markersize": 3,
            "alpha": 0.8,
        },
        "Betweenness Centrality Attack": {
            "color": "#42A5F5",
            "linestyle": "--",
            "linewidth": 2.0,
            "marker": "s",
            "markersize": 3,
            "alpha": 0.8,
        },
        "Random Removal (Baseline)": {
            "color": "#66BB6A",
            "linestyle": "-.",
            "linewidth": 1.8,
            "marker": "^",
            "markersize": 3,
            "alpha": 0.7,
        },
        "GAT Predicted Attack": {
            "color": "#E53935",
            "linestyle": "-",
            "linewidth": 3.5,
            "marker": "D",
            "markersize": 4,
            "alpha": 1.0,
        },
    }

    fig, ax = plt.subplots(figsize=figsize)

    for strategy_name, (fractions, gc_sizes) in results.items():
        gc_normalized = [s / initial_nodes for s in gc_sizes]

        style = style_config.get(strategy_name, {
            "color": "#999", "linestyle": "-", "linewidth": 1.5,
            "marker": ".", "markersize": 3, "alpha": 0.8,
        })

        ax.plot(
            fractions,
            gc_normalized,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markevery=5,
            label=strategy_name,
            alpha=style["alpha"],
        )

    # Styling
    ax.set_xlabel("Fraction of Nodes Removed", fontsize=14, fontweight="bold")
    ax.set_ylabel("Giant Component Size (normalized)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Network Collapse Curve — Classical Metrics vs. GAT Prediction",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)

    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=12)

    # 50% collapse threshold line
    ax.axhline(y=0.5, color="#BDBDBD", linestyle=":", alpha=0.7)
    ax.annotate(
        "50% collapse threshold",
        xy=(0.55, 0.51),
        fontsize=10,
        fontstyle="italic",
        color="#888888",
    )

    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        logger.info(f"4-strategy collapse plot saved to '{output_file}'")
    except IOError as e:
        logger.error(f"Failed to save plot: {e}")
        raise
    finally:
        plt.close(fig)


def print_four_strategy_summary(results: dict, initial_nodes: int) -> None:
    """
    Print summary table comparing all 4 attack strategies.

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy name -> (fractions_removed, gc_sizes).
    initial_nodes : int
        Total number of nodes in the original network.
    """
    threshold = initial_nodes * 0.5

    print("\n" + "=" * 70)
    print("  STAGE 3 — FOUR-STRATEGY ATTACK COMPARISON")
    print("=" * 70)
    print(f"  {'Strategy':<40} {'50% Collapse at':>20}")
    print("-" * 70)

    for strategy, (fractions, gc_sizes) in results.items():
        collapse_fraction = None
        for i, size in enumerate(gc_sizes):
            if size < threshold:
                collapse_fraction = fractions[i]
                break

        if collapse_fraction is not None:
            print(f"  {strategy:<40} {collapse_fraction:>18.1%} removed")
        else:
            print(f"  {strategy:<40} {'Never collapsed':>20}")

    print("=" * 70 + "\n")


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    """
    Main execution pipeline for Stage 3.

    Complete pipeline: network generation → impact scores → feature
    extraction → GAT training → attack simulation → visualization.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: GAT-Based Ecological Node Criticality Prediction")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate the network (from Stage 1)
    # ------------------------------------------------------------------
    logger.info("Step 1/7: Generating scale-free network...")
    G = generate_scale_free_network()
    initial_nodes = G.number_of_nodes()
    metrics = calculate_baseline_metrics(G)
    print(f"\n  Network: {metrics['Total Nodes']} nodes, "
          f"{metrics['Total Edges']} edges\n")

    # ------------------------------------------------------------------
    # Step 2: Compute ground-truth impact scores (labels)
    # ------------------------------------------------------------------
    logger.info("Step 2/7: Computing ground-truth impact scores...")
    impact_scores = compute_impact_scores(G)

    # ------------------------------------------------------------------
    # Step 3: Extract node features
    # ------------------------------------------------------------------
    logger.info("Step 3/7: Extracting node features...")
    features = extract_node_features(G)

    # ------------------------------------------------------------------
    # Step 4: Create PyG Data object
    # ------------------------------------------------------------------
    logger.info("Step 4/7: Creating PyTorch Geometric Data object...")
    data = create_pyg_data(G, features, impact_scores)

    # ------------------------------------------------------------------
    # Step 5: Train GAT model
    # ------------------------------------------------------------------
    logger.info("Step 5/7: Training GAT model...")
    model = train_gat_model(data)

    # ------------------------------------------------------------------
    # Step 6: Run all 4 attack simulations
    # ------------------------------------------------------------------
    logger.info("Step 6/7: Running attack simulations...")

    # Classical metrics
    degree_centrality, betweenness_centrality = compute_centrality_metrics(G)

    results = {}

    # 6a: Degree Centrality Attack
    logger.info("  → Degree Centrality Attack...")
    dc_frac, dc_gc = simulate_attack(G, degree_centrality)
    results["Degree Centrality Attack"] = (dc_frac, dc_gc)

    # 6b: Betweenness Centrality Attack
    logger.info("  → Betweenness Centrality Attack...")
    bc_frac, bc_gc = simulate_attack(G, betweenness_centrality)
    results["Betweenness Centrality Attack"] = (bc_frac, bc_gc)

    # 6c: Random Removal
    logger.info("  → Random Removal...")
    rand_frac, rand_gc = simulate_random_removal(G)
    results["Random Removal (Baseline)"] = (rand_frac, rand_gc)

    # 6d: GAT Predicted Attack
    logger.info("  → GAT Predicted Attack...")
    gat_predictions = get_gat_predictions(model, data)
    gat_frac, gat_gc = simulate_attack(G, gat_predictions)
    results["GAT Predicted Attack"] = (gat_frac, gat_gc)

    # ------------------------------------------------------------------
    # Step 7: Visualization & Summary
    # ------------------------------------------------------------------
    logger.info("Step 7/7: Generating final visualization...")

    print_four_strategy_summary(results, initial_nodes)
    plot_four_strategy_collapse(results, initial_nodes)

    logger.info("Stage 3 completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Stage 3 failed: {e}")
        sys.exit(1)
