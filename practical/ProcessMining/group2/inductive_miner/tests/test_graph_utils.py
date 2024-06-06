from practical.ProcessMining.group2.inductive_miner.src.graph_utils import *


def test_initialize_graph():
    # Test if graph is initialized correctly with and without input
    graph_no_input = Graph(None)
    assert len(graph_no_input.graph) == 0
    graph_with_input = Graph({'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': []})
    assert len(graph_with_input.graph) > 0


def test_add_edge():
    # Test if edge is added correctly
    g = Graph({'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': []})
    g.add_edge('a', 'c')
    assert g.graph == {'a': ['b', 'c'], 'b': ['c', 'd'], 'c': [], 'd': []}


def test_add_node():
    # Test if node is added correctly
    g = Graph({'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': []})
    g.add_node('e')
    assert g.graph == {'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': [], 'e': []}


def test_get_neighbors():
    # Test if neighbors are returned correctly
    g = Graph({'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': []})
    assert g.get_neighbors('c') == []
    assert g.get_neighbors('b') == ['c', 'd']


def test_get_all_nodes():
    # Test if all nodes are returned correctly
    g = Graph({'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': []})
    assert g.get_all_nodes() == ['a', 'b', 'c', 'd']


def test_is_reachable():
    # Test if reachability is checked correctly
    g = Graph({'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': [], 'e': []})
    assert g.is_reachable('a', 'c') == True
    assert g.is_reachable('c', 'a') == False
    assert g.is_reachable('a', 'e') == False


def test_find_components():
    # Test if components are found correctly
    g = Graph({'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': []})
    assert g.find_components() == [['a'], ['b'], ['d'], ['c']]


def test_reverse_graph():
    # Test if graph is reversed correctly
    g = Graph({'a': ['b'], 'b': ['c', 'd'], 'c': [], 'd': [], 'e': []})
    assert g.reverse_graph() == {'a': [], 'b': ['a'], 'c': ['b'], 'd': ['b'], 'e': []}


def test_dfs():
    # Test if DFS is implemented correctly
    pass


def test_find_non_reachable_pairs():
    # Test if there are non reachable pairs
    pass
