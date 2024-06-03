from typing import List, Set

from practical.ProcessMining.group1.task3.inductiveminer import InductiveMiner, CutType


class TestInductiveMiner:
    def test_parallel_cut(self):
        log = [('b', 'c'),
               ('c', 'b')]
        miner = InductiveMiner(log)
        parallel_cut = miner._parallel_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert check_lists_of_sets_equal(parallel_cut, [set('b'), set('c')])  # order does not matter
        parallel_split = miner._split_log(miner.event_log, parallel_cut)
        sublogs = [sorted([('b',), ('b',)]),
                   sorted([('c',), ('c',)])]
        assert all(sorted(sl) in sublogs for sl in parallel_split)

        log = [('b', 'e'),
               ('b', 'e', 'c', 'd', 'b'),
               ('b', 'c', 'e', 'd', 'b'),
               ('b', 'c', 'd', 'e', 'b'),
               ('e', 'b', 'c', 'd', 'b')]
        miner = InductiveMiner(log)
        parallel_cut = miner._parallel_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert check_lists_of_sets_equal(parallel_cut, [set('e'), set('bcd')])  # order does not matter
        parallel_split = miner._split_log(miner.event_log, parallel_cut)
        sublogs = [sorted([('e',), ('e',), ('e',), ('e',), ('e',)]),
                   sorted([('c', 'd', 'b'), ('b', 'c', 'd'), ('b', 'c', 'd')])]
        assert all(sorted(sl) in sublogs for sl in parallel_split)

    def test_loop_cut(self):
        log = [('b', 'c'),
               ('c', 'b'),
               ('b', 'c', 'e', 'f', 'b', 'c'),
               ('c', 'b', 'e', 'f', 'b', 'c'),
               ('b', 'c', 'e', 'f', 'c', 'b'),
               ('c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b')]
        miner = InductiveMiner(log)
        loop_cut = miner._loop_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert loop_cut == [set('bc'), set('ef')]  # order does matter
        loop_split = miner._split_log(miner.event_log, loop_cut)
        sublogs = [sorted([('b', 'c'), ('c', 'b'), ('b', 'c'), ('b', 'c'), ('c', 'b'), ('b', 'c'), ('b', 'c'),
                           ('c', 'b'), ('c', 'b'), ('b', 'c'), ('c', 'b')]),
                   sorted([('e', 'f'), ('e', 'f'), ('e', 'f'), ('e', 'f'), ('e', 'f')])]
        assert all(sorted(sl) in sublogs for sl in loop_split)

    def test_handle_base_case(self):
        # exactly once
        log = [('a',), ('a',), ('a',)]
        miner = InductiveMiner(log)
        base_cut, operator = miner._handle_base_cases(log)
        assert base_cut == [set('a')]
        assert operator == CutType.NONE
        # never or once
        log = [('',), ('a',), ('a',)]
        miner = InductiveMiner(log)
        base_cut, operator = miner._handle_base_cases(log)
        assert check_lists_of_sets_equal(base_cut, [set('a'), set('𝜏')])  # order does not matter
        assert operator == CutType.XOR
        # once or more than once
        log = [('a',), ('a', 'a'), ('a', 'a', 'a')]
        miner = InductiveMiner(log)
        base_cut, operator = miner._handle_base_cases(log)
        assert base_cut == [set('a'), set('𝜏')]  # order does matter
        assert operator == CutType.LOOP
        # never, once or more than once
        log = [('',), ('a',), ('a', 'a')]
        miner = InductiveMiner(log)
        base_cut, operator = miner._handle_base_cases(log)
        assert base_cut == [set('𝜏'), set('a')]  # order does matter
        assert operator == CutType.LOOP

    def test_fall_through_flower_model(self):
        log = [('a', 'b', 'c', 'd', 'e', 'f', 'g')]
        miner = InductiveMiner(log)
        miner.run()
        assert miner.process_tree_str == f'{CutType.LOOP.value}({InductiveMiner.TAU}, a, b, c, d, e, f, g)'


def check_lists_of_sets_equal(list1: List[Set[str]], list2: List[Set[str]]) -> bool:
    sorted_list1 = sorted([tuple(sorted(s)) for s in list1])
    sorted_list2 = sorted([tuple(sorted(s)) for s in list2])

    return sorted_list1 == sorted_list2
