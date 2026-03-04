"""Legacy script: overlay_network_visualization."""
from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

import logging


logger = logging.getLogger(__name__)
OUTPUT_DIR = 'Evidence_Analysis'


def load_and_match_pathways(n_pairs=20):
    """Load top N gene pairs from INDRA and find matching OmniPath pathways"""
    logger.info("Loading top %s gene pairs from INDRA...", n_pairs)

    # Load INDRA and get top N pairs
    indra_df = pd.read_csv('indra_3hop_no_hgnc_prefix.csv')
    indra_clean = indra_df[np.isfinite(indra_df['pval']) & np.isfinite(indra_df['logfoldchange'])].copy()
    indra_top = indra_clean.nsmallest(n_pairs, 'pval').copy()

    # Create source-target pair identifier
    indra_top['pair'] = indra_top['source'] + '→' + indra_top['target']

    logger.info("Top %s INDRA gene pairs selected", n_pairs)
    logger.info("Sample pairs: %s", list(indra_top['pair'].head(3)))

    # Load OmniPath
    logger.info("\nLoading OmniPath data...")
    omnipath_df = pd.read_csv('omnipath_3hop_synthetic.csv')
    omnipath_df['pair'] = omnipath_df['source'] + '→' + omnipath_df['target']

    # Find matching pairs in OmniPath
    target_pairs = set(indra_top['pair'])
    omnipath_matched = omnipath_df[omnipath_df['pair'].isin(target_pairs)].copy()

    logger.info("\nMatching results:")
    logger.info("  INDRA pairs: %s", len(indra_top))
    logger.info("  OmniPath matches found: %s", len(omnipath_matched))
    logger.info("  Pairs with NO OmniPath data: %s", len(target_pairs) - len(set(omnipath_matched['pair'])))

    # Show which pairs have matches
    matched_pairs = set(omnipath_matched['pair'])
    unmatched_pairs = target_pairs - matched_pairs
    if unmatched_pairs:
        logger.info("\n  Unmatched pairs (first 5): %5]", list(unmatched_pairs)[)

    return indra_top, omnipath_matched


def is_gene_identifier(identifier):
    """Check if identifier is a gene"""
    if pd.isna(identifier):
        return False
    if identifier.startswith('fplx:'):
        return True
    if any(identifier.startswith(prefix) for prefix in ['mesh:', 'go:', 'chebi:', 'uniprot:']):
        return False
    return True


def build_matched_network(indra_df, omnipath_df):
    """Build network comparing pathways for the SAME source→target pairs"""
    G = nx.DiGraph()

    # Track edge sources and pathway details
    edge_sources = {}  # (node1, node2) -> set(['indra', 'omnipath'])
    edge_pathway_info = {}  # (node1, node2) -> {database, pval, logfc, pair}
    pathway_comparison = {}  # pair -> {'indra': [intermediates], 'omnipath': [intermediates]}

    source_targets = set()
    intermediates = set()

    # Process INDRA pathways
    logger.info("\nProcessing INDRA pathways...")
    for _, row in indra_df.iterrows():
        source = row['source']
        target = row['target']
        pair = row['pair']
        int1 = row.get('intermediate_1', None)
        int2 = row.get('intermediate_2', None)
        pval = row['pval']
        logfc = row['logfoldchange']

        if not is_gene_identifier(int1):
            continue
        if pd.notna(int2) and not is_gene_identifier(int2):
            continue

        source_targets.add(source)
        source_targets.add(target)

        # Track pathway for this pair
        if pair not in pathway_comparison:
            pathway_comparison[pair] = {'indra': [], 'omnipath': []}

        if pd.notna(int1):
            pathway_comparison[pair]['indra'].append(int1)
            edge1 = (source, int1)
            G.add_edge(*edge1)
            edge_sources[edge1] = edge_sources.get(edge1, set())
            edge_sources[edge1].add('indra')
            edge_pathway_info[edge1] = {'database': 'INDRA', 'pval': pval, 'logfc': logfc, 'pair': pair}
            intermediates.add(int1)

            if pd.notna(int2):
                pathway_comparison[pair]['indra'].append(int2)
                edge2 = (int1, int2)
                edge3 = (int2, target)
                G.add_edge(*edge2)
                G.add_edge(*edge3)
                edge_sources[edge2] = edge_sources.get(edge2, set())
                edge_sources[edge2].add('indra')
                edge_sources[edge3] = edge_sources.get(edge3, set())
                edge_sources[edge3].add('indra')
                edge_pathway_info[edge2] = {'database': 'INDRA', 'pval': pval, 'logfc': logfc, 'pair': pair}
                edge_pathway_info[edge3] = {'database': 'INDRA', 'pval': pval, 'logfc': logfc, 'pair': pair}
                intermediates.add(int2)
            else:
                edge2 = (int1, target)
                G.add_edge(*edge2)
                edge_sources[edge2] = edge_sources.get(edge2, set())
                edge_sources[edge2].add('indra')
                edge_pathway_info[edge2] = {'database': 'INDRA', 'pval': pval, 'logfc': logfc, 'pair': pair}

    # Process OmniPath pathways (for the SAME pairs)
    logger.info("Processing OmniPath pathways for matching gene pairs...")
    for _, row in omnipath_df.iterrows():
        source = row['source']
        target = row['target']
        pair = row['pair']
        int1 = row.get('intermediate_1', None)
        int2 = row.get('intermediate_2', None)

        if not is_gene_identifier(int1):
            continue
        if pd.notna(int2) and not is_gene_identifier(int2):
            continue

        source_targets.add(source)
        source_targets.add(target)

        # Track pathway for this pair
        if pair not in pathway_comparison:
            pathway_comparison[pair] = {'indra': [], 'omnipath': []}

        if pd.notna(int1):
            pathway_comparison[pair]['omnipath'].append(int1)
            edge1 = (source, int1)
            G.add_edge(*edge1)
            edge_sources[edge1] = edge_sources.get(edge1, set())
            edge_sources[edge1].add('omnipath')
            edge_pathway_info[edge1] = {'database': 'OmniPath', 'pair': pair}
            intermediates.add(int1)

            if pd.notna(int2):
                pathway_comparison[pair]['omnipath'].append(int2)
                edge2 = (int1, int2)
                edge3 = (int2, target)
                G.add_edge(*edge2)
                G.add_edge(*edge3)
                edge_sources[edge2] = edge_sources.get(edge2, set())
                edge_sources[edge2].add('omnipath')
                edge_sources[edge3] = edge_sources.get(edge3, set())
                edge_sources[edge3].add('omnipath')
                edge_pathway_info[edge2] = {'database': 'OmniPath', 'pair': pair}
                edge_pathway_info[edge3] = {'database': 'OmniPath', 'pair': pair}
                intermediates.add(int2)
            else:
                edge2 = (int1, target)
                G.add_edge(*edge2)
                edge_sources[edge2] = edge_sources.get(edge2, set())
                edge_sources[edge2].add('omnipath')
                edge_pathway_info[edge2] = {'database': 'OmniPath', 'pair': pair}

    intermediates = intermediates - source_targets

    # Analyze pathway agreement
    logger.info("\n" + "=" * 60)
    logger.info("PATHWAY COMPARISON FOR SAME GENE PAIRS")
    logger.info("=" * 60)

    pairs_with_both = 0
    pairs_same_intermediates = 0
    pairs_different_intermediates = 0

    for pair, pathways in pathway_comparison.items():
        has_indra = len(pathways['indra']) > 0
        has_omnipath = len(pathways['omnipath']) > 0

        if has_indra and has_omnipath:
            pairs_with_both += 1
            indra_ints = set(pathways['indra'])
            omni_ints = set(pathways['omnipath'])

            if indra_ints == omni_ints:
                pairs_same_intermediates += 1
            else:
                pairs_different_intermediates += 1

    logger.info("\nGene pairs analyzed: %s", len(pathway_comparison))
    logger.info("  Pairs with data from BOTH databases: %s", pairs_with_both)
    logger.info("  Pairs using SAME intermediates: %s", pairs_same_intermediates)
    logger.info("  Pairs using DIFFERENT intermediates: %s", pairs_different_intermediates)

    if pairs_with_both > 0:
        agreement_rate = 100 * pairs_same_intermediates / pairs_with_both
        logger.info("  Agreement rate: %.1f%%", agreement_rate)

    # Edge overlap statistics
    indra_only_edges = sum(1 for sources in edge_sources.values() if sources == {'indra'})
    omnipath_only_edges = sum(1 for sources in edge_sources.values() if sources == {'omnipath'})
    shared_edges = sum(1 for sources in edge_sources.values() if sources == {'indra', 'omnipath'})

    logger.info("\nNetwork statistics:")
    logger.info("  Total nodes: %s", G.number_of_nodes())
    logger.info("  Total edges: %s", G.number_of_edges())
    logger.info("  Source/target genes: %s", len(source_targets))
    logger.info("  Intermediate proteins: %s", len(intermediates))
    logger.info("\nEdge overlap:")
    logger.info("  INDRA only: %s", indra_only_edges)
    logger.info("  OmniPath only: %s", omnipath_only_edges)
    logger.info("  Shared (same edge in both): %s", shared_edges)
    if G.number_of_edges() > 0:
        logger.info("  Overlap percentage: %.1f%%", 100 * shared_edges / G.number_of_edges())

    # Show some example comparisons
    logger.info("\n" + "=" * 6")
    logger.info("EXAMPLE PATHWAY COMPARISONS (first 5 pairs with both databases):")
    logger.info("=" * 60)
    count = 0
    for pair, pathways in sorted(pathway_comparison.items()):
        if len(pathways['indra']) > 0 and len(pathways['omnipath']) > 0:
            count += 1
            if count <= 5:
                indra_path = ' → '.join(pathways['indra'])
                omni_path = ' → '.join(pathways['omnipath'])
                match = " MATCH" if set(pathways['indra']) == set(pathways['omnipath']) else " DIFFERENT"
                logger.info("\n%s", pair)
                logger.info("  INDRA:    %s", indra_path)
                logger.info("  OmniPath: %s", omni_path)
                logger.info("  %s", match)

    return G, source_targets, intermediates, edge_sources, edge_pathway_info, pathway_comparison


def create_interactive_visualization(G, source_targets, intermediates, edge_sources, edge_pathway_info):
    """Create interactive visualization comparing pathways for same gene pairs"""

    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    # Create edge traces by source
    edge_traces = {'indra': [], 'omnipath': [], 'both': []}
    edge_colors = {'indra': 'rgba(65,105,225,0.6)', 'omnipath': 'rgba(255,140,0,0.6)', 'both': 'rgba(148,0,211,0.9)'}
    edge_widths = {'indra': 2, 'omnipath': 2, 'both': 3.5}

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        sources = edge_sources.get(edge, set())
        if sources == {'indra', 'omnipath'}:
            edge_type = 'both'
        elif 'indra' in sources:
            edge_type = 'indra'
        else:
            edge_type = 'omnipath'

        # Enhanced hover text
        hover_text = f"<b>{edge[0]} → {edge[1]}</b><br>Database: {edge_type.upper()}"
        if edge in edge_pathway_info:
            info = edge_pathway_info[edge]
            hover_text += f"<br>Gene Pair: {info['pair']}"
            if 'pval' in info:
                hover_text += f"<br>p-value: {info['pval']:.2e}<br>log2FC: {info['logfc']:.2f}"

        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge_widths[edge_type], color=edge_colors[edge_type]),
            hoverinfo='text',
            text=hover_text,
            showlegend=False
        )
        edge_traces[edge_type].append(edge_trace)

    # Legend traces
    legend_traces = []
    for edge_type, color in [('indra', 'royalblue'), ('omnipath', 'darkorange'), ('both', 'darkviolet')]:
        legend_trace = go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name=f'{edge_type.upper()} {"(agreed pathway)" if edge_type == "both" else ""}',
            line=dict(width=edge_widths[edge_type], color=color)
        )
        legend_traces.append(legend_trace)

    # Node data
    node_data = {
        'intermediate': {'x': [], 'y': [], 'text': [], 'size': [], 'labels': []},
        'source_target': {'x': [], 'y': [], 'text': [], 'size': [], 'labels': []}
    }

    for node in G.nodes():
        x, y = pos[node]
        degree = G.degree(node)
        node_type = 'intermediate' if node in intermediates else 'source_target'
        size = 15 + degree * 3

        hover_text = f"<b>{node}</b><br>"
        hover_text += f"Type: {'Intermediate' if node in intermediates else 'Source/Target'}<br>"
        hover_text += f"Connections: {degree}"

        node_data[node_type]['x'].append(x)
        node_data[node_type]['y'].append(y)
        node_data[node_type]['text'].append(hover_text)
        node_data[node_type]['size'].append(size)
        node_data[node_type]['labels'].append(node)

    # Node traces
    intermediate_trace = go.Scatter(
        x=node_data['intermediate']['x'], y=node_data['intermediate']['y'],
        mode='markers+text', name='Intermediate Proteins',
        text=node_data['intermediate']['labels'],
        textposition='top center', textfont=dict(size=10, color='black', family='Arial Black'),
        hovertext=node_data['intermediate']['text'], hoverinfo='text',
        marker=dict(size=node_data['intermediate']['size'], color='lightgray',
                    line=dict(width=2, color='gray'), opacity=0.9)
    )

    source_target_trace = go.Scatter(
        x=node_data['source_target']['x'], y=node_data['source_target']['y'],
        mode='markers+text', name='Source/Target Genes',
        text=node_data['source_target']['labels'],
        textposition='top center', textfont=dict(size=11, color='darkblue', family='Arial Black'),
        hovertext=node_data['source_target']['text'], hoverinfo='text',
        marker=dict(size=node_data['source_target']['size'], color='lightblue',
                    line=dict(width=2.5, color='darkblue'), opacity=0.95)
    )

    # Combine traces
    all_edge_traces = edge_traces['indra'] + edge_traces['omnipath'] + edge_traces['both']
    data = legend_traces + all_edge_traces + [intermediate_trace, source_target_trace]

    # Statistics
    indra_only = sum(1 for s in edge_sources.values() if s == {'indra'})
    omnipath_only = sum(1 for s in edge_sources.values() if s == {'omnipath'})
    shared = sum(1 for s in edge_sources.values() if s == {'indra', 'omnipath'})
    total = indra_only + omnipath_only + shared
    overlap_pct = 100 * shared / total if total > 0 else 0

    stats_text = f"<b>Same Gene Pairs Comparison</b><br>"
    stats_text += f"INDRA only edges: {indra_only}<br>"
    stats_text += f"OmniPath only edges: {omnipath_only}<br>"
    stats_text += f"<b>Agreed edges: {shared}</b><br>"
    stats_text += f"Edge agreement: {overlap_pct:.1f}%<br>"
    stats_text += f"Total edges: {G.number_of_edges()}"

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title=dict(
                text='<b>INDRA vs OmniPath - Same Gene Pairs Pathway Comparison</b><br><sub>Blue=INDRA only | Orange=OmniPath only | Purple=Both Agree</sub>',
                font=dict(size=20), x=0.5, xanchor='center'
            ),
            showlegend=True, hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=100),
            plot_bgcolor='white',
            width=1400, height=1000,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.95)',
                        bordercolor='black', borderwidth=2, font=dict(size=12)),
            annotations=[dict(
                text=stats_text, xref="paper", yref="paper", x=0.98, y=0.02,
                xanchor='right', yanchor='bottom', showarrow=False,
                bgcolor='rgba(255,255,255,0.95)', bordercolor='black', borderwidth=2,
                font=dict(size=12)
            )]
        )
    )

    output_path = f'{OUTPUT_DIR}/network_matched_pairs.html'
    fig.write_html(output_path)
    logger.info("\nVisualization saved: %s", output_path)
    return fig


def create_static_png(G, source_targets, intermediates, edge_sources):
    """Create static PNG"""
    fig, ax = plt.subplots(figsize=(20, 16))
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    # Draw edges by source
    indra_edges = [e for e in G.edges() if edge_sources.get(e, set()) == {'indra'}]
    omnipath_edges = [e for e in G.edges() if edge_sources.get(e, set()) == {'omnipath'}]
    shared_edges = [e for e in G.edges() if edge_sources.get(e, set()) == {'indra', 'omnipath'}]

    if indra_edges:
        nx.draw_networkx_edges(G, pos, edgelist=indra_edges, edge_color='royalblue',
                               alpha=0.5, width=2, ax=ax, arrows=True, arrowsize=15,
                               arrowstyle='->', connectionstyle='arc3,rad=0.1')
    if omnipath_edges:
        nx.draw_networkx_edges(G, pos, edgelist=omnipath_edges, edge_color='darkorange',
                               alpha=0.5, width=2, ax=ax, arrows=True, arrowsize=15,
                               arrowstyle='->', connectionstyle='arc3,rad=0.1')
    if shared_edges:
        nx.draw_networkx_edges(G, pos, edgelist=shared_edges, edge_color='darkviolet',
                               alpha=0.8, width=3.5, ax=ax, arrows=True, arrowsize=20,
                               arrowstyle='->', connectionstyle='arc3,rad=0.1')

    # Draw nodes
    intermediate_nodes = [n for n in G.nodes() if n in intermediates]
    nx.draw_networkx_nodes(G, pos, nodelist=intermediate_nodes,
                           node_color='lightgray', node_size=400, alpha=0.9,
                           edgecolors='gray', linewidths=2, ax=ax)

    source_target_nodes = [n for n in G.nodes() if n in source_targets]
    nx.draw_networkx_nodes(G, pos, nodelist=source_target_nodes,
                           node_color='lightblue', node_size=500,
                           edgecolors='darkblue', linewidths=3, ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold',
                            font_family='sans-serif', ax=ax)

    # Statistics
    indra_only = sum(1 for s in edge_sources.values() if s == {'indra'})
    omnipath_only = sum(1 for s in edge_sources.values() if s == {'omnipath'})
    shared = sum(1 for s in edge_sources.values() if s == {'indra', 'omnipath'})

    stats_text = f"Same Gene Pairs\nINDRA only: {indra_only}\nOmniPath only: {omnipath_only}\nAgreed: {shared}"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='royalblue', lw=3, label='INDRA only'),
        Line2D([0], [0], color='darkorange', lw=3, label='OmniPath only'),
        Line2D([0], [0], color='darkviolet', lw=4, label='Both databases agree')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=13, frameon=True,
              fancybox=True, shadow=True)

    ax.set_title('INDRA vs OmniPath - Same Gene Pairs Pathway Comparison\n' +
                 '(Comparing pathways for the SAME source→target pairs)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    output_path = f'{OUTPUT_DIR}/network_matched_pairs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info("Static PNG saved: %s", output_path)
    plt.close()


def main():
    logger.info("=" * 60)
    logger.info("CORRECT ANALYSIS: Same Gene Pairs Pathway Comparison")
    logger.info("=" * 60)

    # Load top gene pairs from INDRA and find matches in OmniPath
    indra_df, omnipath_df = load_and_match_pathways(n_pairs=20)

    if len(omnipath_df) == 0:
        logger.info("\nWARNING: No matching pathways found in OmniPath!")
        logger.info("This could mean:")
        logger.info("  1. OmniPath doesn't have data for these gene pairs")
        logger.info("  2. The source→target naming doesn't match exactly")
        return

    # Build network comparing pathways for same pairs
    G, source_targets, intermediates, edge_sources, edge_pathway_info, pathway_comparison = build_matched_network(
        indra_df, omnipath_df)

    if G.number_of_nodes() == 0:
        logger.info("ERROR: No valid pathways found!")
        return

    # Create visualizations
    logger.info("\nGenerating visualizations...")
    create_interactive_visualization(G, source_targets, intermediates, edge_sources, edge_pathway_info)
    create_static_png(G, source_targets, intermediates, edge_sources)

    logger.info("\n" + "=" * 60)
    logger.info(" Analysis complete!")
    logger.info("=" * 60)
    logger.info("\nFiles generated:")
    logger.info("  • network_matched_pairs.html (interactive)")
    logger.info("  • network_matched_pairs.png (publication-ready)")
    logger.info("\nThis is now a TRUE comparison - same gene pairs, different databases!")


if __name__ == "__main__":
    main()
