def greedy_vertex_cover(edges):
    cover = set()

    while edges:
        edge = edges.pop()

        cover.add(edge[0])
        cover.add(edge[1])

        edges = [e for e in edges if e[0] not in edge and e[1] not in edge]
    return cover


edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 7), (5, 6), (5, 7), (6, 7)]
r = greedy_vertex_cover(edges)
print(r)
