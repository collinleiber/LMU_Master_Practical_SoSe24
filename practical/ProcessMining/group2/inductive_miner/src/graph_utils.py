from typing import List, Optional
from collections import defaultdict
import copy

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
    
    def get_all_edges(self) -> list:
        edges = [(node,neighbor) for node in self.graph for neighbor in self.graph[node]]
        return edges

    def __str__(self) -> str:
        result = ""
        for node in self.graph:
            result += f"{node} -> {self.graph[node]}\n"
        return result
    
    @staticmethod
    def build_graph_from_edges(edges : set) -> 'Graph':
        graph = {}

        for u, v in edges:
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            graph[u].append(v)
            graph[v].append(u)
        return Graph(graph)

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
    
    def convert_to_undirected(self) -> 'Graph':
        undirected = copy.deepcopy(self.graph)
        for node in self.graph.keys():
            for neighbor in self.graph[node]:
                if node not in self.graph[neighbor]:
                    undirected[neighbor].append(node)
        return Graph(undirected)

    def find_components(self) -> List[List]:
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        visited = set()
        components = []

        for node in self.graph.keys():
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)

        return components

    def find_strongly_con_components(self) -> List[List]:
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
    
    def build_cuts_graph(self, cuts) -> 'Graph':
        cut_graph = {k: set() for k in range(len(cuts))}
        cut_map = {}

        for i, scc in enumerate(cuts):
            for node in scc:
                cut_map[node] = i

        for node in self.graph:
            neighbors = self.get_neighbors(node)
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if (neighbor1 != neighbor2): 
                        if (cut_map[node] == cut_map[neighbor1]):
                            if (cut_map[node] != cut_map[neighbor2]):
                                cut_graph[cut_map[node]].add(cut_map[neighbor2])
                        elif (cut_map[neighbor1] == cut_map[neighbor2]):
                            cut_graph[cut_map[node]].add(cut_map[neighbor1])
                        elif (self.is_reachable(neighbor1, neighbor2)):
                            cut_graph[cut_map[node]].add(cut_map[neighbor1])
                        elif (cut_graph[cut_map[node]] == set):
                            cut_graph[cut_map[node]].add(cut_map[neighbor2])
            if len(neighbors) == 1:
                neighbor = neighbors[0]
                if cut_map[node] != cut_map[neighbor]: 
                    cut_graph[cut_map[node]].add(cut_map[neighbor])


        return Graph(cut_graph), cut_map
    

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

    def find_unreachable_pairs(self): # TODO: refactor
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
    
    def traverse_path(self, start_node):
        traversal_order = []
        current_node = start_node
        while current_node is not None and current_node not in traversal_order:
            traversal_order.append(current_node)
            next_nodes = list(self.graph[current_node])
            current_node = next_nodes[0] if next_nodes else None
        return traversal_order


# graph = {'a': ['b', 'c'], 'b': ['c', 'd', 'e'], 'c': ['d', 'b', 'e'], 'd': [], 'e': ['f'], 'f': ['c', 'b']}
# g = Graph(graph)
# scc = g.find_components()
# print(scc)
