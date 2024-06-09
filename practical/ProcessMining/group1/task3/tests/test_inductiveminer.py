import pm4py
import pytest
import graphviz as gviz
from typing import List, Set, Tuple
from unittest.mock import patch
from practical.ProcessMining.group1.shared.utils import event_log_to_dataframe, check_lists_of_sets_equal
from practical.ProcessMining.group1.task3.inductiveminer import InductiveMiner, CutType
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_vis


class TestInductiveMiner:

    @pytest.mark.parametrize(
        "log",
        [
            ([('a', 'b', 'c', 'd'),
              ('a', 'c', 'b', 'd'),
              ('a', 'b', 'c', 'e', 'f', 'b', 'c', 'd'),
              ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'd'),
              ('a', 'b', 'c', 'e', 'f', 'c', 'b', 'd'),
              ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b', 'd')]),
        ]
    )
    def test_get_dfg(self, log: List[Tuple[str]]):
        miner = InductiveMiner(log)
        im_result = miner._get_dfg(log)
        pm4py_result = pm4py.discover_dfg(pm4py.format_dataframe(event_log_to_dataframe(log), case_id='case_id',
                                                                 activity_key='activity', timestamp_key='timestamp'))
        assert im_result == pm4py_result

    @pytest.mark.parametrize(
        "log,cut_result,split_result",
        [
            ([('b', 'c'), ('c', 'b')],
             [set('b'), set('c')],
             [sorted([('b',), ('b',)]), sorted([('c',), ('c',)])]),

            ([('b', 'e'),
             ('b', 'e', 'c', 'd', 'b'),
             ('b', 'c', 'e', 'd', 'b'),
             ('b', 'c', 'd', 'e', 'b'),
             ('e', 'b', 'c', 'd', 'b')],
             [set('e'), set('bcd')],
             [sorted([('e',), ('e',), ('e',), ('e',), ('e',)]),
              sorted([('b',),
                      ('b', 'c', 'd', 'b'),
                      ('b', 'c', 'd', 'b'),
                      ('b', 'c', 'd', 'b'),
                      ('b', 'c', 'd', 'b')])])
        ]
    )
    def test_parallel_cut(self, log: List[Tuple[str]], cut_result: List[Set], split_result: List[Tuple]):
        miner = InductiveMiner(log)

        parallel_cut = miner._parallel_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert check_lists_of_sets_equal(parallel_cut, cut_result)  # order does not matter

        parallel_split = miner._split_log(miner.event_log, parallel_cut, CutType.PARALLEL)
        assert all(sorted(sl) in split_result for sl in parallel_split)


    @pytest.mark.parametrize(
        "log,expected_cut,expected_split",
        [
            ([('a', 'b', 'c', 'd'),
              ('a', 'c', 'b', 'd'),
              ('a', 'b', 'd'),
              ('a', 'c', 'd')],
             None, None),

            ([('a', 'b', 'c'),
              ('a', 'c', 'b'),
              ('a', 'b', 'c', 'e', 'f', 'b', 'c'),
              ('a', 'c', 'b', 'e', 'f', 'b', 'c'),
              ('a', 'b', 'c', 'e', 'f', 'c', 'b'),
              ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b')],
             None, None),

            ([('b', 'c', 'd'),
              ('c', 'b', 'd'),
              ('b', 'c', 'e', 'f', 'b', 'c', 'd'),
              ('c', 'b', 'e', 'f', 'b', 'c', 'd'),
              ('b', 'c', 'e', 'f', 'c', 'b', 'd'),
              ('c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b', 'd')],
             [set('bcef'), set('d')], None),

            ([('a', 'b', 'c', 'd'),
              ('a', 'c', 'b', 'd'),
              ('a', 'b', 'c', 'e', 'f', 'b', 'c', 'd'),
              ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'd'),
              ('a', 'b', 'c', 'e', 'f', 'c', 'b', 'd'),
              ('a', 'c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b', 'd')],
             [set('a'), set('bcef'), set('d')],
             [sorted([('a',), ('a',), ('a',), ('a',), ('a',), ('a',)]),
              sorted([('b', 'c'),
                      ('c', 'b'),
                      ('b', 'c', 'e', 'f', 'b', 'c'),
                      ('c', 'b', 'e', 'f', 'b', 'c'),
                      ('b', 'c', 'e', 'f', 'c', 'b'),
                      ('c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b')]),
              sorted([('d',), ('d',), ('d',), ('d',), ('d',), ('d',)])])
        ]
    )
    def test_sequence_cut(self, log: List[Tuple[str]], expected_cut: List[Set], expected_split: List[Tuple]):
        miner = InductiveMiner(log)

        sequence_cut = miner._sequence_cut(miner.dfg, miner.start_activities, miner.end_activities)
        if expected_cut:
            assert sequence_cut == expected_cut  # order does matter
        print('sequence_cut', sequence_cut)

        if expected_split:
            sequence_split = miner._split_log(miner.event_log, sequence_cut, CutType.SEQUENCE)
            assert all(sorted(sl) in expected_split for sl in sequence_split)

    @pytest.mark.parametrize(
        "log,expected_cut,expected_split",
        [
            ([('a', 'c', 'e', 'q'), ('b', 'd', 'f', 'r'), ('a', 'c', 'e'), ('b', 'd', 'f')],
             [set('aceq'), set('bdfr')],
             [sorted([('a', 'c', 'e', 'q'), ('a', 'c', 'e')]),
              sorted([('b', 'd', 'f', 'r'), ('b', 'd', 'f')])]),
            ([('b', 'c'), ('c', 'b'), ('e',)],
             [set('bc'), set('e')],
             [sorted([('b', 'c'), ('c', 'b')]),
              sorted([('e',)])]),

        ]
    )
    def test_xor_cut(self, log: List[Tuple[str]], expected_cut: List[Set], expected_split: List[Tuple]):
        miner = InductiveMiner(log)

        xor_cut = miner._xor_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert check_lists_of_sets_equal(xor_cut, expected_cut)  # order does not matter

        xor_split = miner._split_log(miner.event_log, xor_cut, CutType.XOR)
        assert all(sorted(sl) in expected_split for sl in xor_split)

    @pytest.mark.parametrize(
        "log,expected_cut,expected_split",
        [
            ([('b', 'c'),
              ('c', 'b'),
              ('b', 'c', 'e', 'f', 'b', 'c'),
              ('c', 'b', 'e', 'f', 'b', 'c'),
              ('b', 'c', 'e', 'f', 'c', 'b'),
              ('c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b')],
             [set('bc'), set('ef')],
             [sorted([('b', 'c'), ('c', 'b'), ('b', 'c'), ('b', 'c'), ('c', 'b'), ('b', 'c'), ('b', 'c'),
                      ('c', 'b'), ('c', 'b'), ('b', 'c'), ('c', 'b')]),
              sorted([('e', 'f'), ('e', 'f'), ('e', 'f'), ('e', 'f'), ('e', 'f')])]),
        ]
    )
    def test_loop_cut(self, log: List[Tuple[str]], expected_cut: List[Set], expected_split: List[Tuple]):
        miner = InductiveMiner(log)

        loop_cut = miner._loop_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert loop_cut == expected_cut  # order does matter

        loop_split = miner._split_log(miner.event_log, loop_cut, CutType.LOOP)
        assert all(sorted(sl) in expected_split for sl in loop_split)

    @pytest.mark.parametrize(
        "log,expected_cut,expected_operator",
        [
            ([('a',), ('a',), ('a',)],  # exactly once
             [set('a')], CutType.NONE),
            ([('',), ('a',), ('a',)],  # never or once
             [set('a'), set('𝜏')], CutType.XOR),
            ([('a',), ('a', 'a'), ('a', 'a', 'a')],  # once or more than once
             [set('a'), set('𝜏')], CutType.LOOP),
            ([('',), ('a',), ('a', 'a')],   # never, once or more than once
             [set('𝜏'), set('a')], CutType.LOOP)
        ]
    )
    def test_handle_base_case(self, log: List[Tuple[str]], expected_cut: List[Set], expected_operator: CutType):
        miner = InductiveMiner(log)
        base_cut, operator = miner._handle_base_cases(log)
        assert check_lists_of_sets_equal(base_cut, expected_cut)
        assert operator == expected_operator

    @pytest.mark.parametrize(
        "log,expected_string",
        [
            ([('a', 'b', 'c', 'd'), ('d', 'a', 'b'), ('a', 'd', 'c'), ('b', 'c', 'd',)],
             f'{CutType.LOOP.value}({InductiveMiner.TAU}, a, b, c, d)'),
        ]
    )
    def test_fall_through_flower_model(self, log: List[Tuple[str]], expected_string: str):
        miner = InductiveMiner(log)
        # process_tree = inductive_miner.apply(pm4py.format_dataframe(event_log_to_dataframe(log), case_id='case_id',
        #                                                          activity_key='activity', timestamp_key='timestamp'))
        # print('process_tree', process_tree)
        miner.run()
        assert miner.process_tree_str == expected_string

    @pytest.mark.parametrize(
        "log",
        [
            ([('a', 'b', 'c', 'd', 'e', 'f', 'b', 'd', 'c', 'e', 'g'), ('a', 'b', 'd', 'c', 'e', 'g'), ('a', 'b', 'c', 'd', 'e', 'f', 'b', 'c', 'd', 'e', 'f', 'b', 'd', 'c', 'e', 'g')]),
            ([('a', 'c', 'd'), ('b', 'c', 'd'), ('a', 'c', 'e'),('b', 'c', 'e')])
        ]
    )
    @patch('practical.ProcessMining.group1.task3.inductiveminer.pt_vis.view')
    @patch('practical.ProcessMining.group1.task3.inductiveminer.pt_vis.save')
    def test_visualize_process_tree(self, mock_save, mock_view, log: List[Tuple[str]]):
        miner = InductiveMiner(log)
        miner.run()

        miner.visualize_process_tree()

        # Check if the view method was called
        mock_view.assert_called_once()

        # Optionally check if the save method was called when actually saving the image
        # mock_save.assert_called_once_with(gviz, "process_tree.png")
