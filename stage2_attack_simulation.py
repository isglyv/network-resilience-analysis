"""
Stage 2: Hub Attack Simulation
================================
Simulates network robustness (habitat resilience) through targeted
node removal (attacks) on the scale-free network generated in Stage 1.

Three attack strategies are compared:
    1. Degree Centrality Attack — removes highest-degree nodes first
    2. Betweenness Centrality Attack — removes highest-betweenness nodes first
    3. Random Node Removal — baseline comparison

This module produces the "Network Collapse Curve" showing how the
giant component degrades under each strategy.

Author: Network Resilience Analysis Project
Date: 2026
References:
    Albert, R., Jeong, H., & Barabási, A.-L. (2000). Error and attack
    tolerance of complex networks. Nature, 406(6794), 378-382.
"""

import sys
import copy
import logging
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable

# Import Stage 1 for network generation
from stage1_network_generation import generate_scale_free_network, calculate_baseline_metrics

# =============================================================================
# Configuration & Constants
# =============================================================================

REMOVAL_FRACTION = 0.01   # Remove top 1% of nodes per iteration
RANDOM_SEED = 42          # Seed for random removal reproducibility
COLLAPSE_PLOT_FILE = "network_collapse_simulation.png"
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 8)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Centrality Calculations
# =============================================================================

def compute_centrality_metrics(G: nx.Graph) -> Tuple[Dict, Dict]:
    """
    Compute Degree Centrality and Betweenness Centrality for all nodes.

    Parameters
    ----------
    G : nx.Graph
        The input network graph.

    Returns
    -------
    tuple of (dict, dict)
        - degree_centrality: {node: centrality_value}
        - betweenness_centrality: {node: centrality_value}

    Raises
    ------
    ValueError
        If the graph is empty.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute centrality on an empty graph.")

    logger.info("Computing Degree Centrality...")
    degree_centrality = nx.degree_centrality(G)

    logger.info("Computing Betweenness Centrality (this may take a moment)...")
    betweenness_centrality = nx.betweenness_centrality(G)

    logger.info(f"Centrality metrics computed for {G.number_of_nodes()} nodes.")

    return degree_centrality, betweenness_centrality


# =============================================================================
# Giant Component Calculation
# =============================================================================

def get_giant_component_size(G: nx.Graph) -> int:
    """
    Return the size of the largest connected component (giant component).

    Parameters
    ----------
    G : nx.Graph
        The input network graph.

    Returns
    -------
    int
        Number of nodes in the giant component. Returns 0 if graph is empty.
    """
    if G.number_of_nodes() == 0:
        return 0

    return len(max(nx.connected_components(G), key=len))


# =============================================================================
# Attack Simulation
# =============================================================================

def simulate_attack(
    G: nx.Graph,
    metric_dict: Dict[int, float],
    removal_fraction: float = REMOVAL_FRACTION
) -> Tuple[List[float], List[int]]:
    """
    Simulate a targeted attack by iteratively removing the most central nodes.

    At each iteration, the top `removal_fraction` (1%) of remaining nodes
    (ranked by centrality) are removed, and the giant component size is
    recorded. Centrality values are NOT recalculated after each removal
    to maintain computational efficiency (initial ranking strategy).

    Parameters
    ----------
    G : nx.Graph
        The input network graph (will be copied, not modified in place).
    metric_dict : dict
        Dictionary mapping node -> centrality value.
    removal_fraction : float
        Fraction of nodes to remove per iteration (default: 0.01 = 1%).

    Returns
    -------
    tuple of (List[float], List[int])
        - fractions_removed: cumulative fraction of nodes removed at each step
        - gc_sizes: giant component size at each step
    """
    G_copy = copy.deepcopy(G)
    initial_nodes = G_copy.number_of_nodes()
    nodes_to_remove_per_step = max(1, int(initial_nodes * removal_fraction))

    # Sort nodes by centrality in descending order
    sorted_nodes = sorted(metric_dict.keys(), key=lambda x: metric_dict[x], reverse=True)

    fractions_removed = [0.0]
    gc_sizes = [get_giant_component_size(G_copy)]

    total_removed = 0
    node_index = 0

    while node_index < len(sorted_nodes) and G_copy.number_of_nodes() > 0:
        # Determine nodes to remove in this step
        nodes_batch = []
        count = 0
        while node_index < len(sorted_nodes) and count < nodes_to_remove_per_step:
            node = sorted_nodes[node_index]
            if G_copy.has_node(node):
                nodes_batch.append(node)
                count += 1
            node_index += 1

        if not nodes_batch:
            break

        # Remove the batch of nodes
        G_copy.remove_nodes_from(nodes_batch)
        total_removed += len(nodes_batch)

        # Record metrics
        fraction = total_removed / initial_nodes
        gc_size = get_giant_component_size(G_copy)

        fractions_removed.append(fraction)
        gc_sizes.append(gc_size)

    return fractions_removed, gc_sizes


def simulate_random_removal(
    G: nx.Graph,
    removal_fraction: float = REMOVAL_FRACTION,
    seed: int = RANDOM_SEED
) -> Tuple[List[float], List[int]]:
    """
    Simulate random node removal as a baseline comparison.

    Parameters
    ----------
    G : nx.Graph
        The input network graph (will be copied, not modified in place).
    removal_fraction : float
        Fraction of nodes to remove per iteration.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (List[float], List[int])
        - fractions_removed: cumulative fraction of nodes removed at each step
        - gc_sizes: giant component size at each step
    """
    random.seed(seed)
    G_copy = copy.deepcopy(G)
    initial_nodes = G_copy.number_of_nodes()
    nodes_to_remove_per_step = max(1, int(initial_nodes * removal_fraction))

    fractions_removed = [0.0]
    gc_sizes = [get_giant_component_size(G_copy)]

    total_removed = 0

    while G_copy.number_of_nodes() > 0:
        remaining_nodes = list(G_copy.nodes())
        if not remaining_nodes:
            break

        # Randomly select nodes to remove
        batch_size = min(nodes_to_remove_per_step, len(remaining_nodes))
        nodes_batch = random.sample(remaining_nodes, batch_size)

        G_copy.remove_nodes_from(nodes_batch)
        total_removed += len(nodes_batch)

        fraction = total_removed / initial_nodes
        gc_size = get_giant_component_size(G_copy)

        fractions_removed.append(fraction)
        gc_sizes.append(gc_size)

    return fractions_removed, gc_sizes


# =============================================================================
# Visualization
# =============================================================================

def plot_collapse_curves(
    results: Dict[str, Tuple[List[float], List[int]]],
    initial_nodes: int,
    output_file: str = COLLAPSE_PLOT_FILE,
    figsize: Tuple[int, int] = PLOT_FIGSIZE,
    dpi: int = PLOT_DPI
) -> None:
    """
    Plot the Network Collapse Curve for all attack strategies.

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy name -> (fractions_removed, gc_sizes).
    initial_nodes : int
        Total number of nodes in the original network (for normalization).
    output_file : str
        Filename for the saved plot.
    figsize : tuple
        Figure dimensions (width, height) in inches.
    dpi : int
        Dots per inch for the saved figure.
    """
    # Color and style configuration for each strategy
    style_config = {
        "Degree Centrality Attack": {
            "color": "#E53935",
            "linestyle": "-",
            "linewidth": 2.5,
            "marker": "o",
            "markersize": 3
        },
        "Betweenness Centrality Attack": {
            "color": "#1E88E5",
            "linestyle": "--",
            "linewidth": 2.5,
            "marker": "s",
            "markersize": 3
        },
        "Random Removal (Baseline)": {
            "color": "#43A047",
            "linestyle": "-.",
            "linewidth": 2.0,
            "marker": "^",
            "markersize": 3
        },
    }

    fig, ax = plt.subplots(figsize=figsize)

    for strategy_name, (fractions, gc_sizes) in results.items():
        # Normalize GC sizes to fraction of original
        gc_normalized = [s / initial_nodes for s in gc_sizes]

        style = style_config.get(strategy_name, {
            "color": "#999999", "linestyle": "-", "linewidth": 1.5,
            "marker": ".", "markersize": 3
        })

        ax.plot(
            fractions,
            gc_normalized,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markevery=5,  # Show marker every 5 data points
            label=strategy_name,
            alpha=0.9
        )

    # Labels and styling
    ax.set_xlabel("Fraction of Nodes Removed", fontsize=14, fontweight="bold")
    ax.set_ylabel("Giant Component Size (normalized)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Network Collapse Curve — Targeted vs. Random Node Removal",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)

    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add critical threshold annotation
    ax.axhline(y=0.5, color="#BDBDBD", linestyle=":", alpha=0.7)
    ax.annotate(
        "50% collapse threshold",
        xy=(0.5, 0.51),
        fontsize=10,
        fontstyle="italic",
        color="#888888"
    )

    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        logger.info(f"Collapse curve plot saved to '{output_file}'")
    except IOError as e:
        logger.error(f"Failed to save plot: {e}")
        raise
    finally:
        plt.close(fig)


# =============================================================================
# Summary Statistics
# =============================================================================

def print_attack_summary(
    results: Dict[str, Tuple[List[float], List[int]]],
    initial_nodes: int
) -> None:
    """
    Print a summary table comparing attack strategies.

    Shows the fraction of nodes removed to reach 50% collapse
    (giant component drops below 50% of original size).

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy name -> (fractions_removed, gc_sizes).
    initial_nodes : int
        Total number of nodes in the original network.
    """
    threshold = initial_nodes * 0.5

    print("\n" + "=" * 70)
    print("  ATTACK SIMULATION SUMMARY")
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
    Main execution pipeline for Stage 2.

    Generates the network (via Stage 1), computes centrality metrics,
    runs three attack simulations, and plots the collapse curves.
    """
    logger.info("=" * 55)
    logger.info("STAGE 2: Hub Attack Simulation")
    logger.info("=" * 55)

    # Step 1: Generate the network (reuse Stage 1)
    logger.info("Generating scale-free network from Stage 1...")
    G = generate_scale_free_network()
    initial_nodes = G.number_of_nodes()

    # Print baseline metrics
    metrics = calculate_baseline_metrics(G)
    print("\n  Baseline: {} nodes, {} edges, GC ratio: {}%".format(
        metrics["Total Nodes"],
        metrics["Total Edges"],
        metrics["Giant Component Size Ratio (%)"]
    ))

    # Step 2: Compute centrality metrics
    degree_centrality, betweenness_centrality = compute_centrality_metrics(G)

    # Step 3: Run simulations
    results = {}

    # Simulation 1: Degree Centrality Attack
    logger.info("Running Degree Centrality Attack simulation...")
    dc_fractions, dc_gc = simulate_attack(G, degree_centrality)
    results["Degree Centrality Attack"] = (dc_fractions, dc_gc)
    logger.info(f"  → {len(dc_fractions)} removal steps completed.")

    # Simulation 2: Betweenness Centrality Attack
    logger.info("Running Betweenness Centrality Attack simulation...")
    bc_fractions, bc_gc = simulate_attack(G, betweenness_centrality)
    results["Betweenness Centrality Attack"] = (bc_fractions, bc_gc)
    logger.info(f"  → {len(bc_fractions)} removal steps completed.")

    # Simulation 3: Random Removal (Baseline)
    logger.info("Running Random Removal simulation...")
    rand_fractions, rand_gc = simulate_random_removal(G)
    results["Random Removal (Baseline)"] = (rand_fractions, rand_gc)
    logger.info(f"  → {len(rand_fractions)} removal steps completed.")

    # Step 4: Print summary
    print_attack_summary(results, initial_nodes)

    # Step 5: Plot collapse curves
    plot_collapse_curves(results, initial_nodes)

    logger.info("Stage 2 completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Stage 2 failed: {e}")
        sys.exit(1)
