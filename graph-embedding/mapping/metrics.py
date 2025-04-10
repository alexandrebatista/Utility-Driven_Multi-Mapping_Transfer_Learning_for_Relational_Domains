import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.spatial.distance import euclidean, cosine
from mapping.apply_mappings import read_domain
from sklearn.metrics.pairwise import cosine_similarity


def extract_graph_metrics(domain_data, bin_edges=None):
    if bin_edges is None:
        bin_edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, np.inf]

    nodes_by_type = domain_data["nodes"]
    edges_by_type = domain_data["edges"]

    total_edges = sum(len(edges) for edges in edges_by_type.values())

    total_nodes = 0
    node_type_map = {}
    for ntype, nodes in nodes_by_type.items():
        total_nodes += len(nodes)
        for node in nodes:
            node_type_map[node] = ntype

    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    degs_by_type = defaultdict(lambda: {"in": [], "out": []})
    edge_type_counts = {}

    G = nx.DiGraph()
    G.add_nodes_from(node_type_map)

    for rel_type, edge_set in edges_by_type.items():
        edge_type_counts[rel_type] = len(edge_set)

        for src, tgt in edge_set:
            G.add_edge(src, tgt)
            out_deg[src] += 1
            in_deg[tgt] += 1

    for node, dtype in node_type_map.items():
        degs_by_type[dtype]["in"].append(in_deg[node])
        degs_by_type[dtype]["out"].append(out_deg[node])

    def degree_stats(degree_list):
        if not degree_list:
            return dict(mean=0.0, std=0.0, median=0.0, max=0, p25=0.0, p75=0.0)
        arr = np.array(degree_list)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "max": int(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75))
        }

    def binned_histogram(degree_list, bin_edges):
        if not degree_list:
            return {f"{bin_edges[i]}–{bin_edges[i + 1] - 1}": 0 for i in range(len(bin_edges) - 1)}
        hist = np.histogram(degree_list, bins=bin_edges)[0]
        return {
            f"{bin_edges[i]}–{bin_edges[i + 1] - 1}": int(hist[i])
            for i in range(len(hist))
        }

    metrics = {
        "node_type_distribution": {
            t: len(nodes) / total_nodes
            for t, nodes in nodes_by_type.items()
        },
        "edge_type_distribution": {
            t: c / total_edges
            for t, c in edge_type_counts.items()
        },
        "in_degree_stats": {
            t: degree_stats(v["in"])
            for t, v in degs_by_type.items()
        },
        "out_degree_stats": {
            t: degree_stats(v["out"])
            for t, v in degs_by_type.items()
        },
        "degree_histograms": {
            t: {
                "in": binned_histogram(v["in"], bin_edges),
                "out": binned_histogram(v["out"], bin_edges)
            }
            for t, v in degs_by_type.items()
        },
    }

    # Componentes fortemente conectados
    sccs = nx.strongly_connected_components(G)
    sizes = [len(c) for c in sccs]
    metrics["strongly_connected_components"] = {
        "num_components": len(sizes),
        "largest_component_size": max(sizes) if sizes else 0
    }

    return metrics


def metrics_to_vector(metrics, node_types, edge_types, bin_labels=None):
    if bin_labels is None:
        bin_labels = ["0–0", "1–1", "2–3", "4–7", "8–15", "16–31", "32–63", "64–127", "128–inf"]

    vec = []

    # 1. Distribuição de tipos de nós (ordem fixa)
    for nt in node_types:
        vec.append(metrics["node_type_distribution"].get(nt, 0.0))

    # 2. Distribuição de tipos de arestas
    for et in edge_types:
        vec.append(metrics["edge_type_distribution"].get(et, 0.0))

    # 3. Estatísticas de grau (in e out) por tipo
    for nt in node_types:
        for dir in ["in", "out"]:
            stats = metrics[f"{dir}_degree_stats"].get(nt, {})
            vec.extend([
                stats.get("mean", 0.0),
                stats.get("std", 0.0),
                stats.get("median", 0.0),
                stats.get("max", 0),
                stats.get("p25", 0.0),
                stats.get("p75", 0.0)
            ])

    # 4. Histogramas binned (in e out)
    for nt in node_types:
        for dir in ["in", "out"]:
            bins = metrics["degree_histograms"].get(nt, {}).get(dir, {})
            for label in bin_labels:
                vec.append(bins.get(label, 0))

    # 5. Componentes fortemente conectados
    scc = metrics["strongly_connected_components"]
    vec.append(scc["num_components"])
    vec.append(scc["largest_component_size"])

    return np.array(vec, dtype=np.float32)


def cosine_sim(v1, v2):
    v1 = np.array(v1).reshape(1, -1)
    v2 = np.array(v2).reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]


def compare_graphs(vec1, vec2, metric="euclidean"):
    if metric == "euclidean":
        return euclidean(vec1, vec2)
    elif metric == "cosine":
        return cosine(vec1, vec2)
    elif metric == "manhattan":
        return np.sum(np.abs(vec1 - vec2))
    else:
        raise ValueError(f"Métrica desconhecida: {metric}")


def extract_vector_metrics_from_domain_file(domain_path):
    domain_data = read_domain(domain_path)
    metrics = extract_graph_metrics(domain_data)

    vec = metrics_to_vector(metrics,
                            node_types=list(domain_data["nodes"].keys()),
                            edge_types=list(domain_data["edges"].keys()))
    return vec

def extract_vector_metrics_from_domain_data(domain_data):
    metrics = extract_graph_metrics(domain_data)

    vec = metrics_to_vector(metrics,
                            node_types=list(domain_data["nodes"].keys()),
                            edge_types=list(domain_data["edges"].keys()))
    return vec
