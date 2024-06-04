from typing import List, Optional
from collections import defaultdict

class Graph:
    # TODO: Add type hints and clean up
    def __init__(self, initial_graph: Optional[dict] = None):
        if initial_graph is not None:
            self.graph = initial_graph
        else:
            self.graph = defaultdict(list)  # Adjacency list, format: "node_id" : [children]

    def add_edge(self, u, v) -> None:
        self.graph[u].append(v)

    def add_node(self, u) -> None:
        if u not in self.graph:
            self.graph[u] = []

    def get_neighbors(self, u) -> list:
        return self.graph[u]

    def get_all_nodes(self) -> list:
        return list(self.graph.keys())

    def __str__(self) -> str:
        result = ""
        for node in self.graph:
            result += f"{node} -> {self.graph[node]}\n"
        return result

    def is_reachable(self, node1, node2) -> bool:
        def dfs(current, target, visited):
            if current == target:
                return True
            visited.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    if dfs(neighbor, target, visited):
                        return True
            return False
        
        return dfs(node1, node2, visited = set())

    def find_components(self) -> List[List]:
        # Kosaraju's Algorithm

        def fill_stack(node):
            # Get order for reverse dfs
            visited.add(node)
            for child in self.graph[node]:
                if child not in visited:
                    fill_stack(child)
            stack.append(node) 

        def reverse_graph(self) -> dict:
            # Reverse all edges
            reversed_graph = defaultdict(list)
            for node in self.graph.keys():
                for child in self.graph[node]:
                    reversed_graph[child].append(node)
            return reversed_graph

        def dfs(graph, node, scc):
            scc.append(node)
            visited.add(node)
            for child in graph[node]:
                if child not in visited:
                    dfs(graph, child, scc)

        visited = set()
        stack = []
        sccs = [] # Strongly connected components

        # 1. Apply DFS on original graph and find order of nodes
        for node in self.graph.keys():
            if node not in visited:
                fill_stack(node) 

        # 2. Reverse the graph
        reversed_graph = reverse_graph(self)

        # 3. Apply DFS on reversed graph in order of nodes from stack
        visited = set()
        while stack:
            node = stack.pop()
            if node not in visited:
                scc = []
                dfs(reversed_graph, node, scc)
                sccs.append(scc)

        return sccs
    
    def build_scc_graph(self, sccs):
        scc_graph = {k: set() for k in range(len(sccs))}
        scc_map = {}

        for i, scc in enumerate(sccs):
            for node in scc:
                scc_map[node] = i

        for node in self.graph:
            for neighbor in self.get_neighbors(node):
                if scc_map[node] != scc_map[neighbor]:
                    scc_graph[scc_map[node]].add(scc_map[neighbor])

        return scc_graph
    

    def dfs_in_dag(self, start, visited):
        stack = [start]
        reachable = set()
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                reachable.add(node)
                stack.extend(self.graph.get(node, [])) # TODO: maybe refactor a little bit
        return reachable

    def all_pairs_reachability_dag(self):
        nodes = list(self.graph.keys())
        reach = {node: set() for node in nodes}
        
        for node in nodes:
            visited = set()
            reach[node] = self.dfs_in_dag(node, visited)
        
        return reach

    def find_non_reachable_pairs(self): # TODO: refactor
        reach = self.all_pairs_reachability_dag()
        nodes = list(self.graph.keys())
        non_reachable_pairs = set()
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                if v not in reach[u] and u not in reach[v]:
                    non_reachable_pairs.add((u, v))

        return list(non_reachable_pairs)


# graph = {'a': ['b', 'c'], 'b': ['c', 'd', 'e'], 'c': ['d', 'b', 'e'], 'd': [], 'e': ['f'], 'f': ['c', 'b']}
# g = Graph(graph)
# scc = g.find_components()
# print(scc)
