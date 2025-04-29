import heapq


class TopKMapper:
    def __init__(self, K, mode="max"):
        assert mode in {"max", "min"}, "mode must be 'max' or 'min'"
        self.K = K
        self.mode = mode
        self.heap = []

    def _score(self, similarity):
        # Negamos a similaridade para capturar os menores
        return similarity if self.mode == "max" else -similarity

    def update(self, similarity, mapping_id, mapping_data):
        item = (self._score(similarity), mapping_id, mapping_data)
        if len(self.heap) < self.K:
            heapq.heappush(self.heap, item)
        else:
            if self.mode == "max" and item[0] > self.heap[0][0]:
                heapq.heappushpop(self.heap, item)
            elif self.mode == "min" and item[0] < self.heap[0][0]:
                heapq.heappushpop(self.heap, item)

    def get_mappings(self):
        mappings = [self._restore(item) for item in self.heap]
        return sorted(
            mappings,
            key=lambda x: -x[0] if self.mode == "max" else x[0]
        )

    def _restore(self, item):
        return (item[0], item[1], item[2]) if self.mode == "max" else (-item[0], item[1], item[2])


def rank_by_key(solutions, key, node_type=None, metric="kl_divergences", ranking_name="ranking_kl"):
    # Sort the list according to the given key

    if node_type is None:
        sorted_list = sorted(solutions, key=lambda x: x[metric][key])
    else:
        sorted_list = sorted(solutions, key=lambda x: x[metric][key][node_type])

    # Update each item with its ranking
    for position, item in enumerate(sorted_list, start=1):
        ranking = item.setdefault(ranking_name, {})

        if node_type is None:
            ranking[key] = position
        else:
            # Ensure the ranking for node_type exists and set the position
            degree_ranking = ranking.setdefault(key, {})
            degree_ranking[node_type] = position

def aggregate_ranks(solutions):
    for solution in solutions:
        ranks = ([solution["ranking_kl"]["node_kl"]] +
                 [solution["ranking_kl"]["edge_kl"]] +
                 list(solution["ranking_kl"]["degree_kl_in"].values()) +
                 list(solution["ranking_kl"]["degree_kl_out"].values()))


        solution["mean_ranking_kl"] = sum(ranks) / len(ranks)


