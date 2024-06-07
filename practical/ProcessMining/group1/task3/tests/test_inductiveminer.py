from typing import List, Set
from practical.ProcessMining.group1.shared.utils import event_log_to_dataframe, check_lists_of_sets_equal
from practical.ProcessMining.group1.task3.inductiveminer import InductiveMiner, CutType
import pm4py


class TestInductiveMiner:
    def test_get_dfg(self):
        log = [('a', 'b', 'c', 'd'),
               ('a', 'c', 'b', 'd'),
               ('a', 'b', 'c', 'e', 'f', 'b', 'c', 'd'),
               ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'd'),
               ('a', 'b', 'c', 'e', 'f', 'c', 'b', 'd'),
               ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b', 'd')]
        inductive_miner = InductiveMiner(log)
        im_result = inductive_miner._get_dfg(log)
        print('im_result ====', im_result)
        pm4py_result = pm4py.discover_dfg(pm4py.format_dataframe(event_log_to_dataframe(log), case_id='case_id',
                                                                 activity_key='activity', timestamp_key='timestamp'))
        assert im_result == pm4py_result

    def test_parallel_cut(self):
        log = [('b', 'c'),
               ('c', 'b')]
        miner = InductiveMiner(log)
        parallel_cut = miner._parallel_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert check_lists_of_sets_equal(parallel_cut, [set('b'), set('c')])  # order does not matter
        parallel_split = miner._split_log(miner.event_log, parallel_cut, CutType.PARALLEL)
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
        parallel_split = miner._split_log(miner.event_log, parallel_cut, CutType.PARALLEL)
        sublogs = [sorted([('e',), ('e',), ('e',), ('e',), ('e',)]),
                   sorted([('b',),
                           ('b', 'c', 'd', 'b'),
                           ('b', 'c', 'd', 'b'),
                           ('b', 'c', 'd', 'b'),
                           ('b', 'c', 'd', 'b')])]
        assert all(sorted(sl) in sublogs for sl in parallel_split)

    def test_sequence_cut(self):
        log = [('a', 'b', 'c', 'd'),
               ('a', 'c', 'b', 'd'),
               ('a', 'b', 'c', 'e', 'f', 'b', 'c', 'd'),
               ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'd'),
               ('a', 'b', 'c', 'e', 'f', 'c', 'b', 'd'),
               ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b', 'd')]
        miner = InductiveMiner(log)
        sequence_cut = miner._sequence_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert sequence_cut == [set('a'), set('bcef'), set('d')]  # order does matter
        sequence_split = miner._split_log(miner.event_log, sequence_cut, CutType.SEQUENCE)
        print('dasdfa =====', sequence_split)
        sublogs = [sorted([('a',), ('a',), ('a',), ('a',), ('a',), ('a',)]),
                   sorted([('b', 'c'),
                           ('c', 'b'),
                           ('b', 'c', 'e', 'f', 'b', 'c'),
                           ('c', 'b', 'e', 'f', 'b', 'c'),
                           ('b', 'c', 'e', 'f', 'c', 'b'),
                           ('c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b')]),
                   sorted([('d',), ('d',), ('d',), ('d',), ('d',), ('d',)])]
        assert all(sorted(sl) in sublogs for sl in sequence_split)

    def test_xor_cut(self):
        log = [('a', 'c', 'e'), ('b', 'd', 'f'), ('a', 'c', 'e'), ('b', 'd', 'f')]
        miner = InductiveMiner(log)
        xor_cut = miner._xor_cut(miner.dfg, miner.start_activities, miner.end_activities)
        print('xor_cut1 ==== ', xor_cut)
        assert check_lists_of_sets_equal(xor_cut, [set('ace'), set('bdf')])  # order does not matter
        xor_split = miner._split_log(miner.event_log, xor_cut, CutType.XOR)
        sublogs = [sorted([('a', 'c', 'e'), ('a', 'c', 'e')]), sorted([('b', 'd', 'f'), ('b', 'd', 'f')])]
        assert all(sorted(sl) in sublogs for sl in xor_split)

        log = [('b', 'c'), ('c', 'b'), ('e',)]
        miner = InductiveMiner(log)
        xor_cut = miner._xor_cut(miner.dfg, miner.start_activities, miner.end_activities)
        print('miner.dfg ==== ', miner.dfg) # dfg does't include e, sp...
        print('xor_cut2 ==== ', xor_cut)
        assert check_lists_of_sets_equal(xor_cut, [set('bc'), set('e')])  # order does not matter
        xor_split = miner._split_log(miner.event_log, xor_cut)
        sublogs = [sorted([('b', 'c'), ('c', 'b')]), sorted([('e',)])]
        assert all(sorted(sl) in sublogs for sl in xor_split)

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
        loop_split = miner._split_log(miner.event_log, loop_cut, CutType.LOOP)
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
