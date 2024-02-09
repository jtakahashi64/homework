def main(graph, start):
    # ステップ1: 初期化
    distance = {vertex: float('infinity') for vertex in graph}
    distance[start] = 0

    # ステップ2: リラクゼーション
    for current_vertex in graph:
        for neighbor, weight in graph[current_vertex].items():
            distance_via_vertex = distance[current_vertex] + weight
            if distance_via_vertex < distance[neighbor]:
                distance[neighbor] = distance_via_vertex

    # ステップ3: 負の重みの閉路の検出
    for u in graph:
        for neighbor, weight in graph[current_vertex].items():
            distance_via_vertex = distance[current_vertex] + weight
            if distance_via_vertex < distance[neighbor]:
                print("Graph contains negative weight cycle")
                return None

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
