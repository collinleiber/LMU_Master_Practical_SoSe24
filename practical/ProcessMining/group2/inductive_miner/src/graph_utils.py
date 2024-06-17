from typing import Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import copy

class Graph:
    def __init__(self, initial_graph: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the graph.
        :param initial_graph: Optional initial adjacency list.
        """
        if initial_graph is not None:
            self.graph = initial_graph
        else:
            self.graph = defaultdict(list)

    def add_edge(self, u: str, v: str) -> None:
        """
        Add an edge to the graph.
        :param u: Starting node of the edge.
        :param v: Ending node of the edge.
        """
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def add_node(self, u: str) -> None:
        """
        Add a node to the graph.
        :param u: Node to be added.
        """
        if u not in self.graph:
            self.graph[u] = []

    def get_neighbors(self, u: str) -> List[str]:
        """
        Get the neighbors of a node.
        :param u: Node whose neighbors are to be fetched.
        :return: List of neighbors.
        """
        return self.graph[u]

    def get_all_nodes(self) -> List[str]:
        """
        Get all nodes in the graph.
        :return: List of all nodes.
        """
        return list(self.graph.keys())
    
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """
        Get all edges in the graph.
        :return: List of all edges.
        """
        return [(node, neighbor) for node in self.graph for neighbor in self.graph[node]]
    
    @staticmethod
    def build_graph_from_edges(edges: Set[Tuple[str, str]]) -> 'Graph':
        """
        Build a graph from a set of edges.
        :param edges: Set of edges.
        :return: Graph object.
        """
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        return Graph(graph)
    
    def dfs(self, start: str, 
            visit_func: Optional[Callable[[str], None]] = None, 
            visited: Optional[set] = None) -> List[str]:
        """
        Generalized DFS method.
        :param start: Starting node for DFS.
        :param visit_func: Function to call on each visited node.
        :param condition_func: Condition function to determine if a neighbor should be visited.
        :return: List of visited nodes.
        """
        if visited is None:
            visited = set()
        stack = [start]
        result = []

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                if visit_func:
                    visit_func(node)
                result.append(node)
                for neighbor in self.get_neighbors(node):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return result

    def is_reachable(self, node1: str, node2: str) -> bool:
        """
        Check if node2 is reachable from node1.
        :param node1: Starting node.
        :param node2: Target node.
        :return: True if reachable, False otherwise.
        """
        path = self.dfs(node1)
        return node2 in path
    
    def convert_to_undirected(self) -> 'Graph':
        """
        Convert the graph to an undirected graph.
        :return: Undirected graph.
        """
        undirected = copy.deepcopy(self.graph)
        for node in self.graph.keys():
            for neighbor in self.graph[node]:
                if node not in self.graph[neighbor]:
                    undirected[neighbor].append(node)
        return Graph(undirected)

    def find_components(self) -> List[Set[str]]:
        """
        Find all connected components in the graph.
        :return: List of components.
        """
        visited = set()
        components = []

        for node in self.graph.keys():
            if node not in visited:
                current_component = set()
                self.dfs(node, visit_func=lambda n: current_component.add(n), visited=visited)
                visited.update(current_component)
                components.append(current_component)

        return components

    def find_strongly_connected_components(self) -> List[List[str]]:
        """
        Find all strongly connected components using Kosaraju's algorithm.
        :return: List of strongly connected components.
        """
        def fill_stack(node: str) -> None:
            """Function to fill the stack with nodes in order of completion."""
            visited.add(node)
            for child in self.graph[node]:
                if child not in visited:
                    fill_stack(child)
            stack.append(node)

        def reverse_graph(self) -> 'Graph':
            """Function to reverse the graph."""
            reversed_graph = defaultdict(list)
            for node in self.graph.keys():
                for neighbor in self.graph[node]:
                    reversed_graph[neighbor].append(node)
            return Graph(reversed_graph)

        visited = set()
        stack = []
        sccs = []

        # 1. Apply DFS on original graph and find order of nodes
        for node in self.graph.keys():
            if node not in visited:
                fill_stack(node)
                visited.update(stack)

        # 2. Reverse the graph
        reversed_graph = reverse_graph(self)

        # 3. Apply DFS on reversed graph in order of nodes from stack
        visited.clear()

        while stack:
            node = stack.pop()
            if node not in visited:
                current_scc = set()
                reversed_graph.dfs(node, visit_func=lambda n: current_scc.add(n), visited=visited)
                visited.update(current_scc)
                sccs.append(current_scc)

        return sccs

    def build_cuts_graph(self, cuts: List[List[str]]) -> Tuple['Graph', Dict[str, int]]:
        """
        Builds a graph representing connections between cuts and maps each node to its corresponding cut index.
        :param cuts: List of lists, where each inner list represents a cut.
        :return: Tuple containing the resulting graph (always a path) and a mapping of nodes to their cut indices.
        """
        cut_graph = {i: set() for i in range(len(cuts))}
        cut_map = {node: i for i, scc in enumerate(cuts) for node in scc}

        for node in self.get_all_nodes():
            current_cut = cut_map[node]
            neighbors = set(self.get_neighbors(node))
            
            for neighbor in neighbors:
                neighbor_cut = cut_map[neighbor]
                # If the neighbor belongs to a different cut, add its cut index to the current cut's set
                if current_cut == neighbor_cut:
                    continue
                if len(neighbors) == 1:
                    cut_graph[current_cut].add(neighbor_cut)
                else:
                    # Check for other neighbors
                    for second_neighbor in (neighbors - {neighbor}):
                        second_cut = cut_map[second_neighbor]
                        if current_cut == second_cut:
                            cut_graph[current_cut].add(neighbor_cut)
                        # Ignore neighbors in the same cut
                        if neighbor_cut == second_cut:
                            cut_graph[current_cut].add(neighbor_cut)
                            break
                        # If a second neighbor belongs to a different cut, check for reachability and the non-reachable neighbor
                        if self.is_reachable(neighbor, second_neighbor):
                            cut_graph[current_cut].add(neighbor_cut)
                            break

        return Graph(cut_graph), cut_map

    def all_pairs_reachability_dag(self) -> Dict[str, Set[str]]:
        """
        Compute reachability for all pairs in a DAG.
        :return: Dictionary with reachability sets.
        """
        nodes = list(self.graph.keys())
        reach = {node: set() for node in nodes}
        
        for node in nodes:
            reach[node] = set(self.dfs(node))
        
        return reach

    def find_unreachable_pairs(self) -> List[Tuple[str, str]]:
        """
        Find all pairs of nodes that are not reachable from each other.
        :return: List of unreachable pairs.
        """
        reach = self.all_pairs_reachability_dag()
        nodes = self.get_all_nodes()
        non_reachable_pairs = set()
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                if v not in reach[u] and u not in reach[v]:
                    non_reachable_pairs.add((u, v))

        return list(non_reachable_pairs)

    def traverse_path(self, start_node: str) -> List[str]:
        """
        Traverse the path starting from the given node.
        :param start_node: Starting node.
        :return: List of nodes in the traversal order.
        """
        traversal_order = []
        current_node = start_node
        while current_node is not None and current_node not in traversal_order:
            traversal_order.append(current_node)
            next_nodes = list(self.graph[current_node])
            current_node = next_nodes[0] if next_nodes else None
        return traversal_order
    
    def __str__(self) -> str:
        """
        String representation of the graph.
        :return: String representation.
        """
        return "\n".join(f"{node} -> {neighbors}" for node, neighbors in self.graph.items())
