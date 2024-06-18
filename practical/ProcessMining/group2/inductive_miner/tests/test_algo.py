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
    e0 = EventLog({})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    assert len(dfg0.graph) == 0
    assert len(dfg0.start_nodes) == 0
    assert len(dfg0.end_nodes) == 0

    e1 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
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
    e2 = EventLog({'abca': 3, 'acd': 1})
    dfg2 = DirectlyFollowsGraph(e2)
    dfg2.construct_dfg()
    assert len(dfg2.graph) > 0
    assert dfg2.graph == {'a': ['b', 'c'], 'b': ['c'], 'c': ['a', 'd'], 'd': []}
    assert dfg2.start_nodes == {'a'}
    assert dfg2.end_nodes == {'a', 'd'}


def test_find_exclusive_choice_cut():
    # Test if exclusive choice split is found correctly
    e0 = EventLog({'e': 1, 'bc': 1, 'cb': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    actual = p0.find_exclusive_choice_cut(dfg0)
    target = [['e'], ['b', 'c']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )

    e1 = EventLog({'abcdfedfghabc': 3, 'abcdfeghabc': 2, 'abcijijkabc': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree(e1)

    assert p1.find_exclusive_choice_cut(dfg1) == None

    e2 = EventLog({})
    dfg2 = DirectlyFollowsGraph(e2)
    dfg2.construct_dfg()
    p2 = ProcessTree(e2)
    assert p2.find_exclusive_choice_cut(dfg2) == []

    # TODO: empty list is probably not the best return parameter?


def test_exclusive_choice_split():
    e0 = EventLog({'e': 1, 'bc': 1, 'cb': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    actual = p0.exclusive_choice_split([['e'], ['b', 'c']])
    target = [['e'], ['bc', 'cb']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )


def test_find_sequence_cut():
    # Test if sequence split is found correctly

    e0 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    actual = p0.find_sequence_cut(dfg0)
    target = [['a'], ['b', 'c', 'e'], ['d']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )

    e1 = EventLog({'abcdfedfghabc': 3, 'abcdfeghabc': 2, 'abcijijkabc': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree(e1)
    assert p1.find_sequence_cut(dfg1) == None

    e2 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1, 'ad': 1})
    dfg2 = DirectlyFollowsGraph(e2)
    dfg2.construct_dfg()
    p2 = ProcessTree(e2)
    actual = p2.find_sequence_cut(dfg2)
    target = [['a'], ['e', 'b', 'c'], ['d']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )

    e3 = EventLog({'abc': 1, 'def': 1})
    dfg3 = DirectlyFollowsGraph(e3)
    dfg3.construct_dfg()
    p3 = ProcessTree(e3)
    assert p3.find_sequence_cut(dfg3) == None

    e4 = EventLog({'a': 1, 'b': 1})
    dfg4 = DirectlyFollowsGraph(e4)
    dfg4.construct_dfg()
    p4 = ProcessTree(e4)
    assert p4.find_sequence_cut(dfg4) == None


def test_sequence_split():
    e0 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    actual = p0.sequence_split([['a'], ['b', 'c', 'e'], ['d']])
    target = [['a'], ['e', 'bc', 'cb'], ['d']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )


def test_find_parallel_cut():
    # Test if parallel split is found correctly
    e0 = EventLog({'bc': 1, 'cb': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    assert p0.find_parallel_cut(dfg0) == [{'b'}, {'c'}]

    e1 = EventLog({'abcdfedfghabc': 3, 'abcdfeghabc': 2, 'abcijijkabc': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree(e1)
    assert p1.find_parallel_cut(dfg1) == None

    e2 = EventLog({})
    dfg2 = DirectlyFollowsGraph(e2)
    dfg2.construct_dfg()
    p2 = ProcessTree(e2)
    assert p2.find_parallel_cut(dfg2) == []
    # TODO Empty List


def test_parallel_split():
    e0 = EventLog({'bc': 1, 'cb': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    assert p0.parallel_split([['b'], ['c']]) == [['b'], ['c']]


def test_find_loop_cut():
    # Test if loop cut is found correctly
    e0 = EventLog(
        {'cbefbcefcb': 1, 'cbefbc': 1, 'bc': 1, 'bcefcb': 1, 'cb': 1, 'bcefbc': 1}
    )
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    actual = p0.find_loop_cut(dfg0)
    target = [['b', 'c'], ['e', 'f']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )

    e1 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree(e1)

    assert p1.find_loop_cut(dfg1) == None

    e2 = EventLog({})
    dfg2 = DirectlyFollowsGraph(e2)
    dfg2.construct_dfg()
    p2 = ProcessTree(e2)
    assert p2.find_loop_cut(dfg2) == None

    e3 = EventLog(
        {'cbefbcefcb': 1, 'cbefbc': 1, 'bc': 1, 'bcefcb': 1, 'cbg': 1, 'bcefbc': 1}
    )
    dfg3 = DirectlyFollowsGraph(e3)
    dfg3.construct_dfg()
    p3 = ProcessTree(e3)
    actual = p3.find_loop_cut(dfg3)
    target = [['b', 'c', 'g'], ['e', 'f']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )

    e4 = EventLog(
        {'cbefbcefxcb': 1, 'cbefbc': 1, 'bc': 1, 'bcefcb': 1, 'cb': 1, 'bcefbc': 1}
    )
    dfg4 = DirectlyFollowsGraph(e4)
    dfg4.construct_dfg()
    p4 = ProcessTree(e4)
    actual = p4.find_loop_cut(dfg4)
    target = [['b', 'c'], ['e', 'f', 'x']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )


def test_loop_split():
    e0 = EventLog(
        {'cbefbcefcb': 1, 'cbefbc': 1, 'bc': 1, 'bcefcb': 1, 'cb': 1, 'bcefbc': 1}
    )
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    actual = p0.loop_split([['b', 'c'], ['e', 'f']])
    target = [['cb', 'bc'], ['ef']]
    assert set(frozenset(sub) for sub in actual) == set(
        frozenset(sub) for sub in target
    )


def test_construct_process_tree():
    # Test if process tree is constructed correctly
    def to_frozenset(item):
        if isinstance(item, list):
            return frozenset(to_frozenset(e) for e in item)
        elif isinstance(item, tuple):
            return tuple(to_frozenset(e) for e in item)
        else:
            return item

    e0 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()

    p0 = ProcessTree(e0)
    actual = p0.construct_process_tree()
    target = (
        '->',
        ['a', ('X', [('||', ['c', 'b']), 'e']), 'd'],
    )
    assert to_frozenset(actual) == to_frozenset(target)

    e1 = EventLog({})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()

    p1 = ProcessTree(e1)
    assert p1.construct_process_tree() == 'tau'


def test_mine_process_model():
    e0 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
    i0 = InductiveMiner()
    p0 = i0.mine_process_model(e0)
    # TODO: recheck functionality of mine_process_model


def test_find_base_case():
    e0 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    assert p0.find_base_case() == None

    e1 = EventLog({'a': 1})
    dfg1 = DirectlyFollowsGraph(e1)
    dfg1.construct_dfg()
    p1 = ProcessTree(e1)
    assert p1.find_base_case() == 'a'

    e2 = EventLog({})
    dfg2 = DirectlyFollowsGraph(e2)
    dfg2.construct_dfg()
    p2 = ProcessTree(e2)
    assert p2.find_base_case() == 'tau'

    e3 = EventLog({'': 1})
    dfg3 = DirectlyFollowsGraph(e3)
    dfg3.construct_dfg()
    p3 = ProcessTree(e3)
    assert p3.find_base_case() == 'tau'


def test_dfg__str__():
    e0 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    assert (
        str(dfg0)
        == "Directly Follows Graph: (\n\tGraph: {'a': ['b', 'c', 'e'], 'b': ['c', 'd'], 'c': ['d', 'b'], 'd': [], 'e': ['d']}\n\tStart nodes: {'a'}\n\tEnd nodes: {'d'}\n)"
    )


def test_pt__str__():
    e0 = EventLog({'abcd': 3, 'acbd': 2, 'aed': 1})
    dfg0 = DirectlyFollowsGraph(e0)
    dfg0.construct_dfg()
    p0 = ProcessTree(e0)
    assert str(p0) == "➜(a, x(∧(b, c), e), d)" or "➜(a, x(e, ∧(b, c)), d)"
