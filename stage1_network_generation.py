"""
Stage 1: Scale-Free Network Generation
========================================
Generates a Scale-Free network using the Barabási-Albert (BA) model
to represent landscape connectivity patterns.

This module creates a network with power-law degree distribution,
calculates baseline metrics, and visualizes the degree distribution.

Author: Network Resilience Analysis Project
Date: 2026
References:
    Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in
    random networks. Science, 286(5439), 509-512.
"""

import sys
import logging
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# =============================================================================
# Configuration & Constants
# =============================================================================

# Barabási-Albert model parameters
BA_NUM_NODES = 2000       # Total number of nodes (N)
BA_NUM_EDGES = 3          # Number of edges to attach from new node (m)
BA_SEED = 42              # Random seed for reproducibility

# Plot configuration
PLOT_DPI = 300            # Resolution for saved figures
PLOT_FIGSIZE = (10, 7)    # Figure dimensions in inches
DEGREE_DIST_FILE = "degree_distribution.png"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Network Generation
# =============================================================================

def generate_scale_free_network(
    n: int = BA_NUM_NODES,
    m: int = BA_NUM_EDGES,
    seed: int = BA_SEED
) -> nx.Graph:
    """
    Generate a Scale-Free network using the Barabási-Albert model.

    The BA model produces networks with power-law degree distributions,
    mimicking the heterogeneous connectivity patterns observed in many
    real-world ecological and landscape networks.

    Parameters
    ----------
    n : int
        Number of nodes in the network.
    m : int
        Number of edges to attach from a new node to existing nodes
        via preferential attachment.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        The generated scale-free network.

    Raises
    ------
    ValueError
        If n <= 0 or m <= 0 or m >= n.
    """
    # Input validation
    if n <= 0:
        raise ValueError(f"Number of nodes must be positive, got {n}")
    if m <= 0:
        raise ValueError(f"Number of edges (m) must be positive, got {m}")
    if m >= n:
        raise ValueError(f"m ({m}) must be less than n ({n})")

    logger.info(f"Generating Barabási-Albert network: N={n}, m={m}, seed={seed}")

    try:
        G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
        logger.info(f"Network generated successfully with {G.number_of_nodes()} "
                     f"nodes and {G.number_of_edges()} edges.")
        return G
    except Exception as e:
        logger.error(f"Failed to generate network: {e}")
        raise


# =============================================================================
# Metric Calculation
# =============================================================================

def calculate_baseline_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Calculate and return baseline network metrics.

    Metrics computed:
        - Total Nodes
        - Total Edges
        - Giant Component Size Ratio (percentage)
        - Average Node Degree

    Parameters
    ----------
    G : nx.Graph
        The input network graph.

    Returns
    -------
    dict
        Dictionary containing the computed metrics.

    Raises
    ------
    ValueError
        If the graph is empty (no nodes).
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot calculate metrics on an empty graph.")

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    # Giant (largest connected) component
    largest_cc = max(nx.connected_components(G), key=len)
    giant_component_ratio = (len(largest_cc) / total_nodes) * 100

    # Average degree: sum of all degrees / number of nodes
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)

    metrics = {
        "Total Nodes": total_nodes,
        "Total Edges": total_edges,
        "Giant Component Size Ratio (%)": round(giant_component_ratio, 2),
        "Average Node Degree": round(avg_degree, 4),
    }

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print baseline network metrics in a formatted table.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values.
    """
    print("\n" + "=" * 55)
    print("  BASELINE NETWORK METRICS")
    print("=" * 55)
    for key, value in metrics.items():
        print(f"  {key:<40} {value:>10}")
    print("=" * 55 + "\n")


# =============================================================================
# Visualization
# =============================================================================

def plot_degree_distribution(
    G: nx.Graph,
    output_file: str = DEGREE_DIST_FILE,
    figsize: Tuple[int, int] = PLOT_FIGSIZE,
    dpi: int = PLOT_DPI
) -> None:
    """
    Plot the node degree distribution as a histogram with logarithmic y-axis.

    The log-scale y-axis is essential for visualizing the power-law
    distribution characteristic of scale-free networks.

    Parameters
    ----------
    G : nx.Graph
        The input network graph.
    output_file : str
        Filename for the saved plot.
    figsize : tuple
        Figure dimensions (width, height) in inches.
    dpi : int
        Dots per inch for the saved figure.
    """
    if G.number_of_nodes() == 0:
        logger.warning("Cannot plot degree distribution for an empty graph.")
        return

    degrees = [d for _, d in G.degree()]

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram with appropriate bin count
    max_degree = max(degrees)
    bins = np.arange(0, max_degree + 2) - 0.5  # Center bins on integer values

    ax.hist(
        degrees,
        bins=bins,
        color="#2196F3",
        edgecolor="#1565C0",
        alpha=0.85,
        linewidth=0.5,
        label=f"N={G.number_of_nodes()}, m={BA_NUM_EDGES}"
    )

    # CRITICAL: Logarithmic y-axis for power-law visualization
    ax.set_yscale("log")

    # Labels and styling
    ax.set_xlabel("Node Degree (k)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency (log scale)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Node Degree Distribution — Barabási-Albert Scale-Free Network",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Add annotation for power-law behavior
    ax.annotate(
        "Power-law tail →",
        xy=(max_degree * 0.5, 1),
        fontsize=10,
        fontstyle="italic",
        color="#666666"
    )

    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        logger.info(f"Degree distribution plot saved to '{output_file}'")
    except IOError as e:
        logger.error(f"Failed to save plot: {e}")
        raise
    finally:
        plt.close(fig)


# =============================================================================
# Main Execution
# =============================================================================

def main() -> nx.Graph:
    """
    Main execution pipeline for Stage 1.

    Generates the scale-free network, computes baseline metrics,
    prints them, and plots the degree distribution.

    Returns
    -------
    nx.Graph
        The generated Barabási-Albert scale-free network.
    """
    logger.info("=" * 55)
    logger.info("STAGE 1: Scale-Free Network Generation")
    logger.info("=" * 55)

    # Step 1: Generate the network
    G = generate_scale_free_network()

    # Step 2: Calculate and display baseline metrics
    metrics = calculate_baseline_metrics(G)
    print_metrics(metrics)

    # Step 3: Plot degree distribution
    plot_degree_distribution(G)

    logger.info("Stage 1 completed successfully.")
    return G


if __name__ == "__main__":
    try:
        G = main()
    except Exception as e:
        logger.critical(f"Stage 1 failed: {e}")
        sys.exit(1)
