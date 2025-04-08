import random

def select_mappings(solutions):
    if len(solutions.get_robust_mappings()) <= 10:
        return solutions.get_robust_mappings()
    else:
        return random.sample(solutions.get_robust_mappings(),10)

def random_sampling_selection(solutions):
    if len(solutions.get_robust_mappings()) <= 10:
        return solutions.get_robust_mappings()
    else:
        return random.sample(solutions.get_robust_mappings(),10)


def structural_metrics_selection(solutions):
    import networkx as nx
    from collections import defaultdict, Counter
    import numpy as np
    import json

    def compute_all_metrics(graph: nx.DiGraph, compute_scc: bool = False):
        metrics = {}

        node_type_count = Counter()
        edge_type_count = Counter()
        in_degree_by_type = defaultdict(list)
        out_degree_by_type = defaultdict(list)
        degree_histogram_by_type = defaultdict(Counter)
        type_to_type_matrix = defaultdict(lambda: defaultdict(Counter))

        for node, data in graph.nodes(data=True):
            ntype = data.get("type", "undefined")
            node_type_count[ntype] += 1

        for u, v, data in graph.edges(data=True):
            etype = data.get("type", "undefined")
            edge_type_count[etype] += 1

            u_type = graph.nodes[u].get("type", "undefined")
            v_type = graph.nodes[v].get("type", "undefined")

            u_out = graph.out_degree(u)
            v_in = graph.in_degree(v)

            out_degree_by_type[u_type].append(u_out)
            in_degree_by_type[v_type].append(v_in)

            degree_histogram_by_type[u_type][u_out] += 1
            degree_histogram_by_type[v_type][v_in] += 1

            type_to_type_matrix[etype][u_type][v_type] += 1

        metrics["node_type_distribution"] = dict(node_type_count)
        metrics["edge_type_distribution"] = dict(edge_type_count)

        metrics["in_degree_by_type"] = {
            ntype: {"mean": float(np.mean(degs)), "std": float(np.std(degs))}
            for ntype, degs in in_degree_by_type.items()
        }

        metrics["out_degree_by_type"] = {
            ntype: {"mean": float(np.mean(degs)), "std": float(np.std(degs))}
            for ntype, degs in out_degree_by_type.items()
        }

        metrics["degree_histograms"] = {
            ntype: dict(hist) for ntype, hist in degree_histogram_by_type.items()
        }

        metrics["type_to_type_matrix"] = {
            etype: {src: dict(dests) for src, dests in sources.items()}
            for etype, sources in type_to_type_matrix.items()
        }

        if compute_scc:
            sccs = list(nx.strongly_connected_components(graph))
            metrics["num_strongly_connected_components"] = len(sccs)
            metrics["largest_scc_size"] = max((len(scc) for scc in sccs), default=0)

        return metrics

    # Simular aplicação em vários grafos
    def compute_batch_metrics(graph_list, compute_scc=False):
        return [compute_all_metrics(G, compute_scc=compute_scc) for G in graph_list]

    # Exemplo com 2 grafos
    G1 = nx.DiGraph()
    G1.add_node(1, type="A")
    G1.add_node(2, type="B")
    G1.add_node(3, type="A")
    G1.add_edge(1, 2, type="rel1")
    G1.add_edge(2, 3, type="rel2")
    G1.add_edge(3, 1, type="rel1")

    G2 = nx.DiGraph()
    G2.add_node("x", type="A")
    G2.add_node("y", type="B")
    G2.add_edge("x", "y", type="rel1")

    batch_result = compute_batch_metrics([G1, G2], compute_scc=True)
    json.dumps(batch_result, indent=2)

