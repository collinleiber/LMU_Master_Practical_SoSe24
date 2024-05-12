from alpha_plus_miner import AlphaMinerplus as amp
from sortedcontainers import SortedSet, SortedDict

traces = {
    '1': ['a', 'b', 'c'],
    '2': ['a', 'c', 'd'],
    '3': ['b', 'c', 'd']
}

def test_initiate_alpha_miner():
    # Test if the AlphaMinerplus class can be initiated
    alpha_miner = amp(traces)
    assert alpha_miner is not None
    assert alpha_miner.traces == traces
    
def test_get_transitions():
    # Test if the transitions are extracted correctly
    alpha_miner = amp(traces)
    assert alpha_miner.getInitialTransitions() == SortedSet(['a', 'b'])
    assert alpha_miner.getTransitions() == SortedSet(['a', 'b', 'c', 'd'])
    assert alpha_miner.getFinalTransitions() == SortedSet(['c', 'd'])

def test_get_footprint():
    # Test if the footprint matrix is extracted correctly
    alpha_miner = amp(traces)
    footprint = alpha_miner.getFootprint()
    # TODO - Implement after refactoring the code
    # assert footprint == None

def test_get_pairs():
    # Test if the pairs are extracted correctly
    alpha_miner = amp(traces)
    # TODO - Implement after refactoring the code
    # assert alpha_miner.pairs == []