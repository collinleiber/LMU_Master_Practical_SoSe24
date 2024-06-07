from practical.ProcessMining.group2.inductive_miner.src.algo import *


def test_file_loading():
    # Tests about different test cases, like empty file, file with only one trace, file with multiple traces
    # Test if file is loaded correctly
    e0 = EventLog.from_file('./../data/log_empty.txt')
    assert len(e0.traces) == 0

    e1 = EventLog.from_file('./../data/log_one_trace.txt')
    assert len(e1.traces) == 1
    assert e1.traces == {
        'acdeh': 1,
    }

    e2 = EventLog.from_file('./../data/log_from_paper.txt')
    assert len(e2.traces) > 0
    assert e2.traces == {
        'acdeh': 1,
        'abdeg': 1,
        'adceh': 1,
        'abdeh': 2,
        'acdeg': 1,
        'adceg': 1,
        'acdefdbeh': 1,
        'adbeg': 1,
        'acdefbdeh': 1,
        'acdefbdeg': 1,
        'acdefdbeg': 1,
        'adcefcdeh': 1,
        'adcefdbeh': 1,
        'adcefbdeg': 1,
        'acdefbdefdbeg': 1,
        'adcefdbeg': 1,
        'adcefbdefbdeg': 1,
        'adcefdbefbdeh': 1,
        'adbefbdefdbeg': 1,
        'adcefdbefcdefdbeg': 1,
    }


def test_construct_dfg():
    # Test if DFG is constructed correctly
    e0 = EventLog.from_traces({})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    assert len(dfg0.graph) == 0
    assert len(dfg0.start_nodes) == 0
    assert len(dfg0.end_nodes) == 0

    e1 = EventLog.from_traces({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    assert len(dfg1.graph) > 0
    assert dfg1.graph == {
        'a': ['b', 'c', 'e'],
        'b': ['c', 'd'],
        'c': ['d', 'b'],
        'd': [],
        'e': ['d'],
    }
    assert dfg1.start_nodes == {'a'}
    assert dfg1.end_nodes == {'d'}

    # Test constructing DFG with cyclic traces
    e2 = EventLog.from_traces({'abca': 3, 'acd': 1})
    dfg2 = DirectlyFollowsGraph(e2)
    dfg2.construct_dfg()
    assert len(dfg2.graph) > 0
    assert dfg2.graph == {'a': ['b', 'c'], 'b': ['c'], 'c': ['a', 'd'], 'd': []}
    assert dfg2.start_nodes == {'a'}
    assert dfg2.end_nodes == {'a', 'd'}


def test_find_exclusive_choice_split():
    # TODO: function returns None, even with traces e0 which should have exclusive choice
    # Test if exclusive choice split is found correctly
    e0 = EventLog.from_traces({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree()
    # assert p0.find_exclusive_choice_split(dfg0) == [['b', 'c'], ['e']]

    e1 = EventLog.from_traces({'abcdfedfghabc': 3, 'abcdfeghabc': 2, 'abcijijkabc': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree()
    # TODO: assertion


def test_find_sequence_split():

    # Test if sequence split is found correctly

    e0 = EventLog.from_traces({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree()
    assert p0.find_sequence_split(dfg0) == [['a'], ['b', 'c'], ['d']]
    # TODO: check if this is correct

    e1 = EventLog.from_traces({'abcdfedfghabc': 3, 'abcdfeghabc': 2, 'abcijijkabc': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree()
    assert p1.find_sequence_split(dfg1) == [
        ['a', 'h', 'g', 'f', 'd', 'c', 'b', 'e', 'k', 'j', 'i']
    ]


def test_find_parallel_split():
    # TODO: function gives none for parallel split
    # Test if parallel split is found correctly
    e0 = EventLog.from_traces({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree()

    # assert p0.find_parallel_split(dfg0) == [['b'], ['c']]

    e1 = EventLog.from_traces({'abcdfedfghabc': 3, 'abcdfeghabc': 2, 'abcijijkabc': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree()
    # TODO: assertion
    # assert p1.find_parallel_split(dfg1) == None


def test_find_loop_split():
    # TODO: order of output of find_loop_split should be the same?
    # Test if loop split is found correctly
    e0 = EventLog.from_traces({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree()
    print(p0.find_loop_split(dfg0), '0')
    assert p0.find_loop_split(dfg0) == None

    e1 = EventLog.from_traces({'abcdfedfghabc': 3, 'abcdfeghabc': 2, 'abcijijkabc': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree()
    # assert p1.find_loop_split(dfg1) == [
    #    ['c', 'b', 'a'],
    #    ['d', 'f', 'e', 'g', 'h'],
    #    ['i', 'j', 'k'],
    # ]


def test_construct_process_tree():
    # Test if process tree is constructed correctly
    pass


def test_mine_process_model():
    # Test if process model is mined correctly
    pass
