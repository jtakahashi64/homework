import heapq


def main(graph, start):
    # ステップ1: 初期化
    distance = {vertex: float('infinity') for vertex in graph}
    distance[start] = 0

    pq = [(0, start)]

    while pq:
        # ステップ2: 最も距離が小さい頂点を選択
        current_distance, current_vertex = heapq.heappop(pq)

        # ステップ3: 隣接する頂点への距離を更新
        for neighbor, weight in graph[current_vertex].items():
            distance_via_vertex = current_distance + weight
            if distance_via_vertex < distance[neighbor]:
                distance[neighbor] = distance_via_vertex
                heapq.heappush(pq, (distance_via_vertex, neighbor))

    return distance


# グラフの例
graph = {
    'A': {'B': 1, 'C': 2},
    'B': {'A': 1, 'C': 3, 'D': 1},
    'C': {'A': 2, 'B': 3, 'D': 4},
    'D': {'B': 1, 'C': 4}
}

start_vertex = 'A'

r = main(graph, start_vertex)

print(r)
