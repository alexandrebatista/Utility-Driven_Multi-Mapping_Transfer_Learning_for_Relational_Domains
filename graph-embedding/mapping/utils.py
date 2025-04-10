import heapq


class TopNMapper:
    def __init__(self, N):
        self.N = N
        self.heap = []  # min-heap [(similaridade, mapeamento)]

    def update(self, similarity, mapping_id, mapping_data):
        item = (similarity, mapping_id, mapping_data)
        if len(self.heap) < self.N:
            heapq.heappush(self.heap, item)
        else:
            # Se o novo for melhor que o pior atual, substitui
            if similarity > self.heap[0][0]:
                heapq.heappushpop(self.heap, item)

    def get_top_mappings(self):
        # Retorna em ordem decrescente de similaridade
        return sorted(self.heap, key=lambda x: -x[0])