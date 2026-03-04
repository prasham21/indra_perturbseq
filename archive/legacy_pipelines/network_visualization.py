"""Legacy script: network_visualization."""
from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import logging


logger = logging.getLogger(__name__)
OUTPUT_DIR = 'Evidence_Analysis'


def load_and_filter_top_pathways():
    """Load 3-hop data and get top 100 by p-value"""
    hop3 = pd.read_csv('indra_3hop_cleaned_results.csv')
    hop3_clean = hop3[np.isfinite(hop3['pval']) & np.isfinite(hop3['logfoldchange'])].copy()
    top_100 = hop3_clean.nsmallest(100, 'pval')

    logger.info("Top 100 pathways by p-value:")
    logger.info("P-value range: %.2e to %.2e", top_100['pval'].min(), top_100['pval'].max())

    return top_100


def build_network(top_pathways):
    """Build network graph from pathways"""
    G = nx.DiGraph()

    source_targets = set()
    intermediates = set()

    for _, row in top_pathways.iterrows():
        source = row['source']
        target = row['target']
        int1 = row.get('intermediate_1', None)
        int2 = row.get('intermediate_2', None)

        source_targets.add(source)
        source_targets.add(target)

        if pd.notna(int1):
            G.add_edge(source, int1)
            intermediates.add(int1)

            if pd.notna(int2):
                G.add_edge(int1, int2)
                intermediates.add(int2)
                G.add_edge(int2, target)
            else:
                G.add_edge(int1, target)
        else:
            G.add_edge(source, target)

    intermediates = intermediates - source_targets

    logger.info("\nNetwork: %s nodes, %s edges", G.number_of_nodes(), G.number_of_edges())

    # Print top hubs
    degrees = dict(G.degree())
    top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("\nTop 10 hub proteins:")
    for node, deg in top_hubs:
        logger.info("  %s: %s connections", node, deg)

    return G, source_targets, intermediates


def create_enhanced_interactive_viz(G, source_targets, intermediates):
    """Create beautiful interactive visualization"""

    pos = nx.spring_layout(G, k=1, iterations=100, seed=42)
    degree_centrality = nx.degree_centrality(G)

    # Edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
            hoverinfo='text',
            text=f"{edge[0]} → {edge[1]}",
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Node data
    node_data = {'intermediate': {'x': [], 'y': [], 'text': [], 'size': [], 'labels': []},
                 'source_target': {'x': [], 'y': [], 'text': [], 'size': [], 'labels': []}}

    degrees = dict(G.degree())
    degree_threshold = np.percentile(list(degrees.values()), 80)

    for node in G.nodes():
        x, y = pos[node]
        degree = G.degree(node)
        node_type = 'intermediate' if node in intermediates else 'source_target'
        size = 10 + degree * 2

        hover_text = f"<b>{node}</b><br>Type: {'Intermediate' if node in intermediates else 'Source/Target'}<br>Connections: {degree}<br>Centrality: {degree_centrality[node]:.3f}"

        node_data[node_type]['x'].append(x)
        node_data[node_type]['y'].append(y)
        node_data[node_type]['text'].append(hover_text)
        node_data[node_type]['size'].append(size)
        node_data[node_type]['labels'].append(node if degree >= degree_threshold else '')

    # Node traces
    intermediate_trace = go.Scatter(
        x=node_data['intermediate']['x'], y=node_data['intermediate']['y'],
        mode='markers+text', name='Intermediate',
        text=node_data['intermediate']['labels'],
        textposition='top center', textfont=dict(size=8),
        hovertext=node_data['intermediate']['text'], hoverinfo='text',
        marker=dict(size=node_data['intermediate']['size'], color='lightgray',
                    line=dict(width=1.5, color='gray'), opacity=0.8)
    )

    source_target_trace = go.Scatter(
        x=node_data['source_target']['x'], y=node_data['source_target']['y'],
        mode='markers+text', name='Source/Target',
        text=node_data['source_target']['labels'],
        textposition='top center', textfont=dict(size=9, color='darkblue'),
        hovertext=node_data['source_target']['text'], hoverinfo='text',
        marker=dict(size=node_data['source_target']['size'], color='lightblue',
                    line=dict(width=2, color='darkblue'), opacity=0.9)
    )

    # Create figure
    data = edge_traces + [intermediate_trace, source_target_trace]
    top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    hub_text = "Top Hubs:<br>" + "<br>".join([f"{n}: {d}" for n, d in top_hubs])

    fig = go.Figure(data=data, layout=go.Layout(
        title=dict(text='<b>Top 100 3-hop Pathways</b><br><sub>Hover for details</sub>',
                   font=dict(size=18), x=0.5),
        showlegend=True, hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=80),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[dict(text=hub_text, xref="paper", yref="paper",
                          x=0.98, y=0.02, xanchor='right', yanchor='bottom',
                          showarrow=False, bgcolor='rgba(255,255,255,0.9)',
                          bordercolor='black', borderwidth=1, font=dict(size=10))]
    ))

    output_path = f'{OUTPUT_DIR}/network_enhanced_interactive.html'
    fig.write_html(output_path)
    logger.info("\nHTML saved: %s", output_path)
    fig.show()


def create_static_png(G, source_targets, intermediates):
    """Create static PNG"""
    fig, ax = plt.subplots(figsize=(16, 16))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax, arrows=True)

    intermediate_nodes = [n for n in G.nodes() if n in intermediates]
    nx.draw_networkx_nodes(G, pos, nodelist=intermediate_nodes,
                           node_color='gray', node_size=100, alpha=0.6, ax=ax)

    source_target_nodes = [n for n in G.nodes() if n in source_targets]
    nx.draw_networkx_nodes(G, pos, nodelist=source_target_nodes,
                           node_color='lightblue', node_size=200,
                           edgecolors='darkblue', linewidths=2, ax=ax)

    high_degree = {n: n for n in G.nodes() if G.degree(n) >= 5}
    nx.draw_networkx_labels(G, pos, labels=high_degree, font_size=8, ax=ax)

    ax.set_title('Top 100 3-hop Pathways Network', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    output_path = f'{OUTPUT_DIR}/network_static.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info("PNG saved: %s", output_path)
    plt.close()


def export_csvs(G, source_targets, intermediates):
    """Export CSVs"""
    # Edges
    edges = [{'source': s, 'target': t} for s, t in G.edges()]
    pd.DataFrame(edges).to_csv(f'{OUTPUT_DIR}/network_edges.csv', index=False)

    # Nodes
    nodes = [{'node': n, 'type': 'intermediate' if n in intermediates else 'source_target',
              'degree': G.degree(n)} for n in G.nodes()]
    pd.DataFrame(nodes).to_csv(f'{OUTPUT_DIR}/network_nodes.csv', index=False)

    logger.info("CSVs saved: network_edges.csv, network_nodes.csv")


def main():
    logger.info("Creating network visualization...")
    top_pathways = load_and_filter_top_pathways()
    G, source_targets, intermediates = build_network(top_pathways)

    create_enhanced_interactive_viz(G, source_targets, intermediates)
    create_static_png(G, source_targets, intermediates)
    export_csvs(G, source_targets, intermediates)

    logger.info("\nAll files generated!")


if __name__ == "__main__":
    main()
