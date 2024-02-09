import networkx as nx
from itertools import combinations


def create_graph(edges):
    G = nx.Graph()

    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    return G


def find_mst(G):
    return nx.minimum_spanning_tree(G)


def find_eulerian_tour(MST):
    MG = nx.MultiGraph(MST)

    for u, v, data in MST.edges(data=True):
        MG.add_edge(u, v, weight=data['weight'])

    eulerian_tour = list(nx.eulerian_circuit(MG))

    return eulerian_tour


def apply_shortcuts(eulerian_tour):
    visited = set()
    path = []

    for u, v in eulerian_tour:
        if u not in visited:
            path.append(u)
            visited.add(u)

    path.append(path[0])

    return path


# グラフの辺を定義: (node1, node2, weight)
edges = [
    (0, 1, 1),
    (1, 2, 2),
    (2, 3, 1),
    (3, 0, 4),
    (0, 2, 3),
    (1, 3, 2)
]

# グラフを作成
G = create_graph(edges)

# 最小全域木を求める
MST = find_mst(G)
print("Minimum Spanning Tree:", MST.edges(data=True))

# オイラー閉路を見つける
eulerian_tour = find_eulerian_tour(MST)
print("Eulerian Tour:", eulerian_tour)

# 巡回路を作成
tour = apply_shortcuts(eulerian_tour)
print("TSP Tour:", tour)
