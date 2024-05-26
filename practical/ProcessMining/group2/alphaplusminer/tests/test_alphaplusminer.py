from alpha_plus_miner import AlphaMinerplus
from sortedcontainers import SortedSet, SortedDict
import os 


def test_without_length_2_loops():
    # Test 1 to check if length 1 loops are correctly handled and visualized correctly.
    
    traces = SortedDict({'1': ['a', 'c'], '2': ['a', 'b', 'c'], '3': ['a', 'b', 'b', 'c'], '4': ['a', 'b', 'b', 'b', 'b', 'c']})
    alpha_miner = AlphaMinerplus(traces)
    assert alpha_miner is not None
    assert alpha_miner.traces == traces
    assert alpha_miner.transitions == SortedSet()
    assert alpha_miner.initial_transitions == SortedSet()
    assert alpha_miner.final_transitions == SortedSet()
    assert alpha_miner.relations == SortedDict()
    assert alpha_miner.pairs == []
    assert alpha_miner.maximal_pairs == []
    assert alpha_miner.places == []
    assert alpha_miner.length_one_loops is None
    assert alpha_miner.log_without_length_one_loops is None
    assert alpha_miner.F_L1L is None
    assert alpha_miner.W_minusL1L == SortedDict()
   

    alpha_miner.get_length_one_loops()
    assert alpha_miner.transitions == {'a', 'b', 'c'}
    assert alpha_miner.length_one_loops == SortedSet({'b'})
    alpha_miner.remove_length_one_loops()
    assert alpha_miner.log_without_length_one_loops == {'a', 'c'}
    alpha_miner.get_FL1L()
    assert alpha_miner.F_L1L == SortedSet([('Place_1', 'b'), ('b', 'Place_1')])
    alpha_miner.generate_W_minus_L1L()
    assert alpha_miner.W_minusL1L ==  SortedDict({'1': ['a', 'c'], '2': ['a', 'c'], '3': ['a', 'c'], '4': ['a', 'c']})


    alpha_miner_plus = AlphaMinerplus(alpha_miner.W_minusL1L)
    assert alpha_miner_plus is not None

    alpha_miner_plus.getInitialTransitions() 
    assert alpha_miner_plus.initial_transitions == SortedSet(['a'])
    alpha_miner_plus.getFinalTransitions()
    assert alpha_miner_plus.final_transitions == SortedSet(['c'])
    alpha_miner_plus.get_transitions()
    assert alpha_miner_plus.transitions == SortedSet(['a', 'c'])
    alpha_miner_plus.get_footprint()
    assert alpha_miner_plus.relations == SortedDict({'a': SortedDict({'a': '#', 'c': '->'}), 'c': SortedDict({'a': '<-', 'c': '#'})})

    alpha_miner_plus.getPairs()
    assert alpha_miner_plus.pairs == [('a', 'c'), (('a',), ('c',))]
    alpha_miner_plus.get_maximal_pairs()
    assert alpha_miner_plus.maximal_pairs == [('a', 'c')]

    alpha_miner_plus.add_places()
   
    assert alpha_miner_plus.places == [('input', SortedSet(['a'])), ('a', 'Place_1', 'c'), (SortedSet(['c']), 'output')]

    alpha_miner_plus.visualize(alpha_miner.F_L1L)

def test_with_length_2_loops():
    trace_2 = SortedDict({'1': ['a', 'b', 'e', 'f'], 
                          '2': ['a', 'b', 'e', 'c', 'd', 'b', 'f'], 
                          '3': ['a', 'b', 'c', 'e', 'd', 'b', 'f'], 
                          '4': ['a', 'b', 'c', 'd', 'e', 'b', 'f'], 
                          '5': ['a', 'e', 'b', 'c', 'd', 'b', 'f']})
    
    alpha_miner = AlphaMinerplus(trace_2)
    assert alpha_miner is not None
    assert alpha_miner.traces == trace_2
    assert alpha_miner.transitions == SortedSet()
    assert alpha_miner.initial_transitions == SortedSet()
    assert alpha_miner.final_transitions == SortedSet()
    assert alpha_miner.relations == SortedDict()
    assert alpha_miner.pairs == []
    assert alpha_miner.maximal_pairs == []
    assert alpha_miner.places == []
    assert alpha_miner.length_one_loops is None
    assert alpha_miner.log_without_length_one_loops is None
    assert alpha_miner.F_L1L is None
    assert alpha_miner.W_minusL1L == SortedDict()
    

    alpha_miner.get_length_one_loops()
    assert alpha_miner.transitions == {'a', 'b', 'c', 'd', 'e', 'f'}
    assert alpha_miner.length_one_loops == SortedSet([])
    alpha_miner.remove_length_one_loops()
    assert alpha_miner.log_without_length_one_loops == {'a', 'b', 'c', 'd', 'e', 'f'}
    alpha_miner.get_FL1L()
    assert alpha_miner.F_L1L == SortedSet([])
    alpha_miner.generate_W_minus_L1L()
    assert alpha_miner.W_minusL1L == SortedDict({'1': ['a', 'b', 'e', 'f'], 
                                                 '2': ['a', 'b', 'e', 'c', 'd', 'b', 'f'], 
                                                 '3': ['a', 'b', 'c', 'e', 'd', 'b', 'f'], 
                                                 '4': ['a', 'b', 'c', 'd', 'e', 'b', 'f'], 
                                                 '5': ['a', 'e', 'b', 'c', 'd', 'b', 'f']})


    alpha_miner_plus = AlphaMinerplus(alpha_miner.W_minusL1L)
    assert alpha_miner_plus is not None

    alpha_miner_plus.getInitialTransitions() 
    assert alpha_miner_plus.initial_transitions == SortedSet(['a'])
    alpha_miner_plus.getFinalTransitions()
    assert alpha_miner_plus.final_transitions == SortedSet(['f'])
    alpha_miner_plus.get_transitions()
    assert alpha_miner_plus.transitions == SortedSet(['a', 'b', 'c', 'd', 'e', 'f'])
    alpha_miner_plus.get_footprint()
    assert alpha_miner_plus.relations == SortedDict({'a': SortedDict({'a': '#', 'b': '->', 'c': '#', 'd': '#', 'e': '->', 'f': '#'}), 
                                                     'b': SortedDict({'a': '<-', 'b': '#', 'c': '->', 'd': '<-', 'e': '||', 'f': '->'}), 
                                                     'c': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '->', 'e': '||', 'f': '#'}), 
                                                     'd': SortedDict({'a': '#', 'b': '->', 'c': '<-', 'd': '#', 'e': '||', 'f': '#'}), 
                                                     'e': SortedDict({'a': '<-', 'b': '||', 'c': '||', 'd': '||', 'e': '#', 'f': '->'}), 
                                                     'f': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '#', 'e': '<-', 'f': '#'})}) 

    alpha_miner_plus.getPairs()
    assert alpha_miner_plus.pairs == [('a', 'b'), 
                                      ('a', 'e'), 
                                      ('b', 'c'), 
                                      ('b', 'f'), 
                                      ('c', 'd'), 
                                      ('d', 'b'), 
                                      ('e', 'f'), 
                                      (('a',), ('b',)), 
                                      (('a',), ('e',)), 
                                      (('a', 'd'), ('b',)), 
                                      (('b',), ('c',)), 
                                      (('b',), ('c', 'f')), 
                                      (('b',), ('c', 'f')), 
                                      (('b',), ('f',)), 
                                      (('c',), ('d',)), 
                                      (('d', 'a'), ('b',)), 
                                      (('d',), ('b',)), 
                                      (('e',), ('f',))] 
    alpha_miner_plus.get_maximal_pairs()
    assert alpha_miner_plus.maximal_pairs == [('a', 'e'), 
                                              ('c', 'd'), 
                                              ('e', 'f'), 
                                              (('a', 'd'), ('b',)), 
                                              (('b',), ('c', 'f'))]

    alpha_miner_plus.add_places()
    assert alpha_miner_plus.places == [('input', SortedSet(['a'])), ('a', 'Place_1', 'e'), ('c', 'Place_2', 'd'), ('e', 'Place_3', 'f'), (('a', 'd'), 'Place_4', ('b',)), (('b',), 'Place_5', ('c', 'f')), (SortedSet(['f']), 'output')]
    
    alpha_miner_plus.visualize(alpha_miner.F_L1L)

def test_edge_cases():
    traces = SortedDict({'1': ['a','b','d'],
                         '2': ['a','b','c','b','d'],
                         '3': ['a','b','c','b','c','b','d']})
    alpha_miner = AlphaMinerplus(traces)
    assert alpha_miner is not None
    assert alpha_miner.traces == traces
    assert alpha_miner.transitions == SortedSet()
    assert alpha_miner.initial_transitions == SortedSet()
    assert alpha_miner.final_transitions == SortedSet()
    assert alpha_miner.relations == SortedDict()
    assert alpha_miner.pairs == []
    assert alpha_miner.maximal_pairs == []
    assert alpha_miner.places == []
    assert alpha_miner.length_one_loops is None
    assert alpha_miner.log_without_length_one_loops is None
    assert alpha_miner.F_L1L is None
    assert alpha_miner.W_minusL1L == SortedDict()
   

    alpha_miner.get_length_one_loops()
    assert alpha_miner.transitions == {'a', 'b', 'c', 'd'}
    assert alpha_miner.length_one_loops == SortedSet([])
    alpha_miner.remove_length_one_loops()
    assert alpha_miner.log_without_length_one_loops == {'b', 'a', 'c', 'd'}
    alpha_miner.get_FL1L()
    assert alpha_miner.F_L1L == SortedSet([])
    alpha_miner.generate_W_minus_L1L()
    assert alpha_miner.W_minusL1L ==  SortedDict({'1': ['a', 'b', 'd'], '2': ['a', 'b', 'c', 'b', 'd'], '3': ['a', 'b', 'c', 'b', 'c', 'b', 'd']})


    alpha_miner_plus = AlphaMinerplus(alpha_miner.W_minusL1L)
    assert alpha_miner_plus is not None

    alpha_miner_plus.getInitialTransitions() 
    assert alpha_miner_plus.initial_transitions == SortedSet(['a'])
    alpha_miner_plus.getFinalTransitions()
    assert alpha_miner_plus.final_transitions == SortedSet(['d'])
    alpha_miner_plus.get_transitions()
    assert alpha_miner_plus.transitions == {'d', 'c', 'b', 'a'}
    alpha_miner_plus.get_footprint()
    assert alpha_miner_plus.relations == SortedDict(
                                        {
                                            'a': SortedDict({'a': '#', 'b': '->', 'c': '#', 'd': '#'}), 
                                            'b': SortedDict({'a': '<-', 'b': '#', 'c': '->', 'd': '->'}), 
                                            'c': SortedDict({'a': '#', 'b': '->', 'c': '#', 'd': '#'}), 
                                            'd': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '#'})
                                         })

    alpha_miner_plus.getPairs()
    assert alpha_miner_plus.pairs == [('a', 'b'), 
                                      ('b', 'c'), 
                                      ('b', 'd'), 
                                      ('c', 'b'), 
                                      (('a',), ('b',)), 
                                      (('a', 'c'), ('b',)), 
                                      (('b',), ('c',)), 
                                      (('b',), ('c', 'd')), 
                                      (('b',), ('c', 'd')), 
                                      (('b',), ('d',)), 
                                      (('c', 'a'), ('b',)), 
                                      (('c',), ('b',))]
    alpha_miner_plus.get_maximal_pairs()
    assert alpha_miner_plus.maximal_pairs == [
                                                (('a', 'c'), 
                                                ('b',)), (('b',), 
                                                ('c', 'd'))
                                            ]

    alpha_miner_plus.add_places()
   
    assert alpha_miner_plus.places == [
                                        ('input', SortedSet(['a'])), 
                                        (('a', 'c'), 'Place_1', ('b',)), 
                                        (('b',), 'Place_2', ('c', 'd')), 
                                        (SortedSet(['d']), 'output')
                                    ]

    alpha_miner_plus.visualize(alpha_miner.F_L1L)