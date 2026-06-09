"""
Stage 4: Interactive HTML Network Visualization
=================================================
Creates two interactive HTML visualizations using Pyvis to demonstrate
the ecological network before and after a targeted hub attack.

Output Files:
    - interactive_ecosystem_demo.html     → Intact network with hubs highlighted
    - interactive_ecosystem_attacked.html → Network after top 5% hubs removed

Author: Network Resilience Analysis Project
Date: 2026
"""

import sys
import logging
import networkx as nx
import numpy as np
from pyvis.network import Network

# =============================================================================
# Configuration & Constants
# =============================================================================

# Scaled-down network for smooth browser rendering
VIS_NUM_NODES = 300
VIS_NUM_EDGES = 3
VIS_SEED = 42

# Hub threshold — top 5% by degree centrality
HUB_PERCENTILE = 95

# Color palette
COLOR_HUB = "#FF3D00"             # Bright red-orange for hubs
COLOR_HUB_BORDER = "#D50000"      # Darker red border
COLOR_NORMAL = "#2E7D32"          # Forest green for regular nodes
COLOR_NORMAL_BORDER = "#1B5E20"   # Darker green border
COLOR_EDGE = "#555555"            # Dark gray edges
COLOR_HUB_EDGE = "#FF6E40"        # Orange edges connected to hubs
COLOR_BG = "#222222"              # Dark background

# Size scaling
NODE_SIZE_MIN = 8
NODE_SIZE_MAX = 50
EDGE_WIDTH_DEFAULT = 0.3

# Output files
OUTPUT_INTACT = "interactive_ecosystem_demo.html"
OUTPUT_ATTACKED = "interactive_ecosystem_attacked.html"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Network Generation (Scaled Down)
# =============================================================================

def generate_visualization_network(
    n: int = VIS_NUM_NODES,
    m: int = VIS_NUM_EDGES,
    seed: int = VIS_SEED,
) -> nx.Graph:
    """
    Generate a smaller BA network suitable for browser visualization.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Edges per new node.
    seed : int
        Random seed.

    Returns
    -------
    nx.Graph
        The generated graph.
    """
    logger.info(f"Generating visualization network: N={n}, m={m}, seed={seed}")
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# =============================================================================
# Hub Identification
# =============================================================================

def identify_hubs(G: nx.Graph, percentile: int = HUB_PERCENTILE) -> set:
    """
    Identify hub nodes as the top percentile by degree centrality.

    Parameters
    ----------
    G : nx.Graph
        The input network.
    percentile : int
        Percentile threshold (e.g., 95 = top 5%).

    Returns
    -------
    set
        Set of hub node IDs.
    """
    degree_centrality = nx.degree_centrality(G)
    threshold = np.percentile(list(degree_centrality.values()), percentile)
    hubs = {n for n, c in degree_centrality.items() if c >= threshold}

    logger.info(f"Identified {len(hubs)} hubs (top {100 - percentile}% by degree)")
    return hubs


# =============================================================================
# Pyvis Network Builder
# =============================================================================

def build_pyvis_network(
    G: nx.Graph,
    hubs: set,
    title: str = "Ecosystem Network",
    highlight_removed: bool = False,
) -> Network:
    """
    Build a styled Pyvis network from a NetworkX graph.

    Parameters
    ----------
    G : nx.Graph
        The NetworkX graph to visualize.
    hubs : set
        Set of hub node IDs (for coloring/sizing in intact view).
    title : str
        Title for the visualization.
    highlight_removed : bool
        If True, this is the attacked version (no hubs present).

    Returns
    -------
    Network
        The configured Pyvis network object.
    """
    net = Network(
        notebook=True,
        width="100%",
        height="800px",
        bgcolor=COLOR_BG,
        font_color="white",
        heading=title,
    )

    # Configure Barnes-Hut physics for organic layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=100,
        spring_strength=0.04,
        damping=0.09,
        overlap=0,
    )

    # Calculate degree for node sizing
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0

    def scale_size(degree):
        """Scale node degree to visual size."""
        if max_degree == min_degree:
            return NODE_SIZE_MIN
        ratio = (degree - min_degree) / (max_degree - min_degree)
        return NODE_SIZE_MIN + ratio * (NODE_SIZE_MAX - NODE_SIZE_MIN)

    # Add nodes
    for node in G.nodes():
        degree = degrees[node]
        size = scale_size(degree)
        is_hub = node in hubs

        if is_hub and not highlight_removed:
            # Hub node styling — prominent red
            color = {
                "background": COLOR_HUB,
                "border": COLOR_HUB_BORDER,
                "highlight": {
                    "background": "#FF6D00",
                    "border": COLOR_HUB_BORDER,
                },
            }
            label = f"🔴 Hub {node}"
            shape = "dot"
            border_width = 3
        else:
            # Normal node styling — forest green
            color = {
                "background": COLOR_NORMAL,
                "border": COLOR_NORMAL_BORDER,
                "highlight": {
                    "background": "#43A047",
                    "border": COLOR_NORMAL_BORDER,
                },
            }
            label = f"{node}"
            shape = "dot"
            border_width = 1

        hover_text = (
            f"<b>Habitat ID:</b> {node}<br>"
            f"<b>Connections:</b> {degree}<br>"
            f"<b>Type:</b> {'🔴 Critical Hub' if is_hub and not highlight_removed else '🟢 Regular Habitat'}"
        )

        net.add_node(
            node,
            label=label,
            title=hover_text,
            size=size,
            color=color,
            shape=shape,
            borderWidth=border_width,
            font={"size": 10, "color": "white", "strokeWidth": 0},
        )

    # Add edges
    for u, v in G.edges():
        u_hub = u in hubs and not highlight_removed
        v_hub = v in hubs and not highlight_removed

        if u_hub or v_hub:
            edge_color = COLOR_HUB_EDGE
            width = 1.0
        else:
            edge_color = COLOR_EDGE
            width = EDGE_WIDTH_DEFAULT

        net.add_edge(u, v, color=edge_color, width=width)

    return net


def inject_custom_html(filepath: str, is_attacked: bool = False) -> None:
    """
    Inject a custom info banner and styling into the generated HTML file.

    Parameters
    ----------
    filepath : str
        Path to the HTML file to modify.
    is_attacked : bool
        Whether this is the attacked version.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()

    if is_attacked:
        banner_color = "#E53935"
        banner_icon = "⚠️"
        banner_title = "ATTACKED NETWORK — Top 5% Hubs Removed"
        banner_desc = (
            "The critical hub habitats have been destroyed. "
            "Notice how the network has shattered into isolated fragments."
        )
    else:
        banner_color = "#2E7D32"
        banner_icon = "🌿"
        banner_title = "INTACT ECOSYSTEM NETWORK"
        banner_desc = (
            "Healthy scale-free network. Red nodes = critical hubs (top 5%). "
            "Hover over nodes for details."
        )

    custom_css_and_banner = f"""
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .info-banner {{
            background: linear-gradient(135deg, {banner_color}CC, {banner_color}88);
            color: white;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            gap: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            position: relative;
        }}
        .info-banner .icon {{
            font-size: 32px;
        }}
        .info-banner .text h2 {{
            margin: 0 0 4px 0;
            font-size: 18px;
            letter-spacing: 1px;
        }}
        .info-banner .text p {{
            margin: 0;
            font-size: 13px;
            opacity: 0.9;
        }}
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            border: 1px solid #444;
            border-radius: 8px;
            padding: 12px 16px;
            color: white;
            font-size: 13px;
            z-index: 1000;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 4px 0;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}
    </style>

    <div class="info-banner">
        <div class="icon">{banner_icon}</div>
        <div class="text">
            <h2>{banner_title}</h2>
            <p>{banner_desc}</p>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item">
            <span class="legend-dot" style="background: {COLOR_HUB};"></span>
            <span>Critical Hub (top 5%)</span>
        </div>
        <div class="legend-item">
            <span class="legend-dot" style="background: {COLOR_NORMAL};"></span>
            <span>Regular Habitat</span>
        </div>
    </div>
    """

    # Inject after <body> tag
    html = html.replace("<body>", f"<body>{custom_css_and_banner}", 1)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Custom HTML banner injected into '{filepath}'")


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    """
    Main pipeline for Stage 4.

    Generates two interactive HTML files:
        1. Intact ecosystem with hubs highlighted
        2. Attacked ecosystem with hubs removed
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Interactive HTML Network Visualization")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate the network
    # ------------------------------------------------------------------
    G = generate_visualization_network()

    # ------------------------------------------------------------------
    # Step 2: Identify hubs
    # ------------------------------------------------------------------
    hubs = identify_hubs(G)

    hub_degrees = {n: G.degree(n) for n in hubs}
    sorted_hubs = sorted(hub_degrees.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 55)
    print("  TOP CRITICAL HUBS")
    print("=" * 55)
    print(f"  {'Node ID':<15} {'Degree':<15} {'Status'}")
    print("-" * 55)
    for node, deg in sorted_hubs[:10]:
        print(f"  {node:<15} {deg:<15} 🔴 Hub")
    print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # Step 3: Build intact network visualization
    # ------------------------------------------------------------------
    logger.info("Building intact ecosystem visualization...")
    net_intact = build_pyvis_network(
        G, hubs, title="🌿 Intact Ecosystem — Scale-Free Network"
    )
    net_intact.save_graph(OUTPUT_INTACT)
    inject_custom_html(OUTPUT_INTACT, is_attacked=False)
    logger.info(f"Intact visualization saved: {OUTPUT_INTACT}")

    # ------------------------------------------------------------------
    # Step 4: Build attacked network visualization
    # ------------------------------------------------------------------
    logger.info("Building attacked ecosystem visualization...")

    # Create attacked graph — remove hub nodes
    G_attacked = G.copy()
    G_attacked.remove_nodes_from(hubs)

    # Stats after attack
    num_components = nx.number_connected_components(G_attacked)
    if G_attacked.number_of_nodes() > 0:
        largest_cc = len(max(nx.connected_components(G_attacked), key=len))
    else:
        largest_cc = 0

    print("=" * 55)
    print("  ATTACK IMPACT")
    print("=" * 55)
    print(f"  Nodes removed:          {len(hubs)}")
    print(f"  Remaining nodes:        {G_attacked.number_of_nodes()}")
    print(f"  Remaining edges:        {G_attacked.number_of_edges()}")
    print(f"  Connected components:   {num_components}")
    print(f"  Largest component:      {largest_cc} nodes")
    print(f"  Fragmentation ratio:    {num_components / G.number_of_nodes() * 100:.1f}%")
    print("=" * 55 + "\n")

    net_attacked = build_pyvis_network(
        G_attacked,
        hubs=set(),  # No hubs in attacked view
        title="⚠️ Attacked Ecosystem — Top 5% Hubs Removed",
        highlight_removed=True,
    )
    net_attacked.save_graph(OUTPUT_ATTACKED)
    inject_custom_html(OUTPUT_ATTACKED, is_attacked=True)
    logger.info(f"Attacked visualization saved: {OUTPUT_ATTACKED}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 55)
    print("  STAGE 4 COMPLETE")
    print("=" * 55)
    print(f"  ✅ {OUTPUT_INTACT}")
    print(f"  ✅ {OUTPUT_ATTACKED}")
    print(f"  Open these files in a web browser to explore!")
    print("=" * 55 + "\n")

    logger.info("Stage 4 completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Stage 4 failed: {e}")
        sys.exit(1)
