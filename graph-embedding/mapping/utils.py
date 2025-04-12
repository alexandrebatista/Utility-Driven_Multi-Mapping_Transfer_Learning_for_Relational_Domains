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
            if item[0] > self.heap[0][0]:
                heapq.heappushpop(self.heap, item)

    def get_mappings(self):
        # Para 'min', invertemos o score de volta ao positivo
        reverse_score = lambda x: x if self.mode == "max" else (-x[0], x[1], x[2])
        return sorted(
            [reverse_score(item) for item in self.heap],
            key=lambda x: -x[0] if self.mode == "max" else x[0]
        )