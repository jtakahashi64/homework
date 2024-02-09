class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u in self.graph:
            self.graph[u].append(v)
        else:
            self.graph[u] = [v]

    def dfs(self, graph, visited, s):
        child_graph = graph.get(s)

        if child_graph is None:
            return

        for g in child_graph:
            visited.append(g)
            self.dfs(graph, visited, g)

        return visited

    def bfs(self, graph, visited, s):
        child_nodes = []
        child_graph = graph.get(s)

        if child_graph is None:
            return

        for g in child_graph:
            visited.append(g)
            child_nodes.append(g)

        for l in child_nodes:
            self.bfs(graph, visited, l)

        return visited


g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(0, 3)
g.add_edge(2, 4)
g.add_edge(3, 5)

r_bfs = g.bfs(g.graph, [], 0)

print(r_bfs)

r_dfs = g.dfs(g.graph, [], 0)

print(r_dfs)
