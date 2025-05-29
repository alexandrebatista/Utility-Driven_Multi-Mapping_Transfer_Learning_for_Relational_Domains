import numpy as np
from collections import defaultdict, OrderedDict
from scipy.special import rel_entr
from sklearn.metrics.pairwise import cosine_similarity


def james_stein_lambda(theta_ml, targets, n):
    """
    Computes the James-Stein shrinkage factor lambda*.

    Args:
        theta_ml (np.ndarray): MLE estimates of shape (p,)
        targets (np.ndarray): Target values t_k of shape (p,)
        n (int): Sample size

    Returns:
        float: Shrinkage factor lambda*
    """
    theta_ml = np.asarray(theta_ml)
    targets = np.asarray(targets)

    numerator = 1 - np.sum(theta_ml ** 2)
    denominator = (n - 1) * np.sum((targets - theta_ml) ** 2)

    if denominator == 0:
        return 0.0  # Avoid division by zero

    lambda_star = numerator / denominator
    return lambda_star



def smooth_distribution(frequencies):
    """
    Computes a smoothed probability distribution using a convex combination
    of the empirical distribution and the uniform distribution.

    Args:
        frequencies (dict): A dictionary where keys are node names and values are counts.
        lambda_ (float): Weight for the empirical distribution. (1 - lambda_) is used for the uniform.

    Returns:
        dict: A dictionary mapping each node to its smoothed probability.
    """
    total_count = sum(frequencies.values())
    num_nodes = len(frequencies)

    # Empirical distribution (normalize frequencies)
    empirical = {
        node: (count / total_count if total_count > 0 else 0.0)
        for node, count in frequencies.items()
    }

    # Uniform distribution (equal probability for all nodes)
    uniform = {node: 1 / num_nodes for node in frequencies}

    lambda_ = james_stein_lambda(np.array(list(empirical.values())), np.array(list(uniform.values())), total_count)

    # Convex combination (smoothed distribution)
    smoothed = {
        node: lambda_ * empirical[node] + (1 - lambda_) * uniform[node]
        for node in frequencies
    }

    return smoothed


def extract_graph_metrics(domain_data, domain_modes, bin_edges=None):
    if bin_edges is None:
        bin_edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, np.inf]

    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    degs_by_type = defaultdict(lambda: {"in": [], "out": []})

    nodes_by_type = OrderedDict()
    for node_types in domain_modes.values():
        for type in node_types:
            if type not in nodes_by_type:
                nodes_by_type[type] = set()

    edge_type_counts = OrderedDict()

    total_edges = 0
    for rel_type, edge_set in domain_data.items():
        total_edges += len(edge_set)
        edge_type_counts[rel_type] = len(edge_set)

        for edge in edge_set:
            if len(edge) == 1:
                src = tgt = edge[0]
                src_type = tgt_type = domain_modes[rel_type][0]
            else:
                src, tgt = edge
                src_type, tgt_type = domain_modes[rel_type]
            # TODO: Add support to ternary relations. Chatgpt suggestion: replace the ternary relation per a new node and 3 binary relations

            out_deg[src] += 1
            in_deg[tgt] += 1
            nodes_by_type[src_type].update([src])
            nodes_by_type[tgt_type].update([tgt])

    total_nodes = 0
    for ntype, nodes in nodes_by_type.items():
        total_nodes += len(nodes)
        degs_by_type[ntype]["in"] += [in_deg[node] for node in nodes]
        degs_by_type[ntype]["out"] += [out_deg[node] for node in nodes]

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
            return OrderedDict(
                (f"{bin_edges[i]}-{bin_edges[i + 1] - 1 if bin_edges[i + 1] != float('inf') else 'inf'}", 0) for i in range(len(bin_edges) - 1)
            )

        hist = np.histogram(degree_list, bins=bin_edges)[0]

        if hist.sum() != 0:
            hist = hist / hist.sum()

        return OrderedDict(
            (f"{bin_edges[i]}-{bin_edges[i + 1] - 1 if bin_edges[i + 1] != float('inf') else 'inf'}", hist[i])
            for i in range(len(hist))
        )

    metrics = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "node_type_distribution": smooth_distribution({key: len(value) for key, value in nodes_by_type.items()}),
        "edge_type_distribution": smooth_distribution(edge_type_counts),
        # "in_degree_stats": {
        #     t: degree_stats(v["in"])
        #     for t, v in degs_by_type.items()
        # },
        # "out_degree_stats": {
        #     t: degree_stats(v["out"])
        #     for t, v in degs_by_type.items()
        # },
        # "degree_histograms": {
        #     t: {
        #         "in": binned_histogram(v["in"], bin_edges),
        #         "out": binned_histogram(v["out"], bin_edges)
        #     }
        #     for t, v in degs_by_type.items()
        # },
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

    return np.array(vec, dtype=np.float32)


def cosine_sim(v1, v2):
    v1 = np.array(v1).reshape(1, -1)
    v2 = np.array(v2).reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]


# def KL(target, candidate, eps=1e-10):
#     a = np.asarray(target, dtype=np.float32) + eps
#     b = np.asarray(candidate, dtype=np.float32) + eps
#     return np.sum(a * np.log(a / b))

def KL(target, candidate):
    return sum(rel_entr(target, candidate))

def compare_graphs(target_metrics, candidate_metrics):
    def get_stats(stat_name):
        target_stat = []
        candidate_stat = []

        for key, value in target_metrics[stat_name].items():
            target_stat.append(value)
            if key not in candidate_metrics[stat_name]:
                pass
            candidate_stat.append(candidate_metrics[stat_name][key])

        return target_stat, candidate_stat

    node_kl = KL(*get_stats("node_type_distribution"))

    edge_kl = KL(*get_stats("edge_type_distribution"))

    # degree_kl_in = {}
    # degree_kl_out = {}
    # for label in target_metrics["degree_histograms"]:
    #     degree_kl_in[label] = KL(list(target_metrics["degree_histograms"][label]["in"].values()),
    #                              list(candidate_metrics["degree_histograms"][label]["in"].values()))
    #     degree_kl_out[label] = KL(list(target_metrics["degree_histograms"][label]["out"].values()),
    #                              list(candidate_metrics["degree_histograms"][label]["out"].values()))

    kl_divergences = {
        "node_kl": node_kl.item(),
        "edge_kl": edge_kl.item(),
        # "degree_kl_in": {label: degree_kl_in[label].item() for label in degree_kl_in},
        # "degree_kl_out": {label: degree_kl_out[label].item() for label in degree_kl_out}
    }

    return kl_divergences


def extract_vector_metrics_from_domain_data(domain_data, domain_modes):
    metrics = extract_graph_metrics(domain_data, domain_modes)
    node_types = set([x for xs in list(domain_modes.values()) for x in xs])
    predicates = list(domain_modes.keys())
    vec = metrics_to_vector(metrics,
                            node_types=node_types,
                            edge_types=predicates)
    return vec
