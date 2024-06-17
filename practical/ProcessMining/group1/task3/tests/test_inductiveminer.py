import pytest
import graphviz as gviz
from typing import List, Set, Tuple
from unittest.mock import patch
from practical.ProcessMining.group1.shared.utils import event_log_to_dataframe, check_lists_of_sets_equal, \
    extract_traces_from_text
from practical.ProcessMining.group1.task3.inductiveminer import InductiveMiner, CutType
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_vis
from IPython.display import Image, display
from unittest.mock import MagicMock, patch


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
            ([('a', 'b', 'a', 'b', 'a'),
              ('a', 'c', 'a', 'c', 'a')],
             [set('a'), set('b'), set('c')],
             [sorted([('a',), ('a',), ('a',), ('a',), ('a',), ('a',)]),
              sorted([('b',), ('b',)]),
              sorted([('c',), ('c',)])]),
        ]
    )
    def test_loop_cut(self, log: List[Tuple[str]], expected_cut: List[Set], expected_split: List[Tuple]):
        miner = InductiveMiner(log)

        loop_cut = miner._loop_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert loop_cut[0] == expected_cut[0]  # order does matter
        assert check_lists_of_sets_equal(loop_cut[1:], expected_cut[1:])

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
            ([('',), ('a',), ('a', 'a')],  # never, once or more than once
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

    @patch('graphviz.Digraph.render', return_value='process_tree.png')
    @patch('graphviz.Digraph.view')
    def test_visualize_process_tree(self, mock_view, mock_render):
        log = [('a', 'b'), ('b', 'c')]
        miner = InductiveMiner(log)
        miner.process_tree_str = '→(×(a, 𝜏), b, ×(c, 𝜏))'  # mock the process tree string

        with patch('IPython.display.Image', wraps=Image) as mock_image:
            with patch('IPython.display.display', wraps=display) as mock_display:
                print("Calling visualize_process_tree...")
                miner.visualize_process_tree()
                mock_render.assert_called_once_with('process_tree', format='png', cleanup=True)
                mock_view.assert_called_once_with('process_tree')
                print(f"Render call args: {mock_render.call_args}")
                print(f"View call args: {mock_view.call_args}")
                print(f"Display call count: {mock_display.call_count}")
                print(f"Image call count: {mock_image.call_count}")
                assert mock_display.call_count == 0
                assert mock_image.call_count == 0

    @patch('pm4py.visualization.petri_net.visualizer.view')
    @patch('pm4py.visualization.petri_net.visualizer.apply', return_value='gviz')
    @patch('pm4py.objects.conversion.process_tree.converter.apply',
           return_value=(MagicMock(), MagicMock(), MagicMock()))
    @patch('pm4py.objects.conversion.log.converter.apply')
    def test_build_and_visualize_petrinet(self, mock_log_converter, mock_pt_converter, mock_pn_apply, mock_pn_view):
        log = [('a', 'b'), ('b', 'c')]
        miner = InductiveMiner(log)
        miner.net = None
        miner.initial_marking = None
        miner.final_marking = None

        # Mocking log converter
        mock_log_converter.return_value = MagicMock()

        # Call the method
        miner.build_and_visualize_petrinet()

        # Assertions
        mock_log_converter.assert_called_once()
        mock_pt_converter.assert_called_once()
        mock_pn_apply.assert_called_once_with(miner.net, miner.initial_marking, miner.final_marking)
        mock_pn_view.assert_called_once()

    @pytest.mark.parametrize(
        "log,expected_tree",
        [
            ("L1 = [<a,b,c,d>^3,<a,c,b,d>^2,<a,e,d>^1]",
             f'{CutType.SEQUENCE.value}(a, {CutType.XOR.value}({CutType.PARALLEL.value}(b, c), e), d)'),
            ("L2 = [<a,b,c,d>^3,<a,c,b,d>^4,<a,b,c,e,f,b,c,d>^2,<a,b,c,e,f,c,b,d>^1,<a,c,b,e,f,b,c,d>^2,<a,c,b,e,f,b,c,e,f,c,b,d>^1]",
                f'{CutType.SEQUENCE.value}(a, {CutType.LOOP.value}({CutType.PARALLEL.value}(b, c), {CutType.SEQUENCE.value}(e, f)), d)'),
            ("L3 = [<a,b,c,d,e,f,b,d,c,e,g>^1,<a,b,d,c,e,g>^2,<a,b,c,d,e,f,b,c,d,e,f,b,d,c,e,g>^1]",
             f'{CutType.SEQUENCE.value}(a, {CutType.LOOP.value}({CutType.SEQUENCE.value}(b, {CutType.PARALLEL.value}(c, d), e), f), g)'),
            ("L4 = [<a,c,d>^45,<b,c,d>^42,<a,c,e>^38,<b,c,e>^22]",
             f'{CutType.SEQUENCE.value}({CutType.XOR.value}(a, b), c, {CutType.XOR.value}(d, e))'),
            ("L5 = [<a,b,e,f>^2,<a,b,e,c,d,b,f>^3,<a,b,c,e,d,b,f>^2,<a,b,c,d,e,b,f>^4,<a,e,b,c,d,b,f>^3]",
             f'{CutType.SEQUENCE.value}(a, {CutType.PARALLEL.value}({CutType.LOOP.value}(b, {CutType.SEQUENCE.value}(c, d)), e), f)'),
            ("L6 = [<a,c,e,g>^2,<a,e,c,g>^3,<b,d,f,g>^2,<b,f,d,g>^4]",
             f'{CutType.SEQUENCE.value}({CutType.XOR.value}({CutType.SEQUENCE.value}(a, {CutType.PARALLEL.value}(c, e)), {CutType.SEQUENCE.value}(b, {CutType.PARALLEL.value}(d, f))), g)'),
            ("L7 = [<a,c>^2,<a,b,c>^3,<a,b,b,c>^2,<a,b,b,b,b,c>^1]",
             f'{CutType.SEQUENCE.value}(a, {CutType.LOOP.value}({InductiveMiner.TAU}, b), c)'),
            ("L8 = [<a,b,d>^3,<a,b,c,b,d>^2,<a,b,c,b,c,b,d>^1]",
             f'{CutType.SEQUENCE.value}(a, {CutType.LOOP.value}(b, c), d)'),
            ("L9 = [<a,c,d>^45,<b,c,e>^42]",
             f'{CutType.SEQUENCE.value}({CutType.XOR.value}(a, b), c, {CutType.XOR.value}(d, e))'),
            ("L10 = [<a,a>^55]",
             f'{CutType.LOOP.value}(a, {InductiveMiner.TAU})'),
            ("L11 = [<a,b,c>^20,<a,c>^30]",
             f'{CutType.SEQUENCE.value}(a, {CutType.XOR.value}(b, {InductiveMiner.TAU}), c)'),
            ("L12 = [<a,c,d>^45,<b,c,e>^42,<a,c,e>^20]",
             f'{CutType.SEQUENCE.value}({CutType.XOR.value}(a, b), c, {CutType.XOR.value}(d, e))'),
            ("L13 = [<a,b,c,d,e>^10,<a,d,b,e>^10,<a,e,b>^1,<a,c,b>^1, <a,b,d,e,c>^2]",
             f'{CutType.SEQUENCE.value}(a, {CutType.PARALLEL.value}(b, {CutType.LOOP.value}({InductiveMiner.TAU}, c, d, e)))'),
            ("L14 = [<a,b,c,d>^10,<d,a,b>^10,<a,d,c>^10,<b,c,d>^5]",
             f'{CutType.LOOP.value}({InductiveMiner.TAU}, a, b, c, d)'),
            ("L15 = [<a,b>^25,<a,c>^25,<d,b>^25,<d,c>^25,<a,b,a,c>^1]",
             f'{CutType.SEQUENCE.value}({CutType.XOR.value}(d, {InductiveMiner.TAU}), {CutType.XOR.value}({CutType.PARALLEL.value}({CutType.LOOP.value}({InductiveMiner.TAU}, a), {CutType.XOR.value}(b, {InductiveMiner.TAU})), {InductiveMiner.TAU}), {CutType.XOR.value}(c, {InductiveMiner.TAU}))'),
            ("L16 = [<a,b,c,d>^20,<a,d>^20]",
             f'{CutType.SEQUENCE.value}(a, {CutType.XOR.value}(b, {InductiveMiner.TAU}), {CutType.XOR.value}(c, {InductiveMiner.TAU}), d)'),
            ("L17 = [<a,b,c,d,e>^10,<a,d,b,e>^10,<a,e,b>^10,<a,c,b>^10,<a,b,d,e,c>^10,<c,a,d>^1]",
             f'{CutType.LOOP.value}({InductiveMiner.TAU}, a, b, c, d, e)'),
            ("L18 = [<a,b,c,g,e,h>^10,<a,b,c,f,g,h>^10,<a,b,d,g,e,h>^10,<a,b,d,e,g>^10,<a,c,b,e,g,h>^10,<a,c,b,f,g,h>^10,<a,d,b,e,g,h>^10,<a,d,b,f,g,h>^10]",
             f'{CutType.SEQUENCE.value}(a, {CutType.PARALLEL.value}(b, {CutType.XOR.value}(c, d)), {CutType.XOR.value}(f, {InductiveMiner.TAU}), {CutType.PARALLEL.value}({CutType.XOR.value}(e, {InductiveMiner.TAU}), g), {CutType.XOR.value}(h, {InductiveMiner.TAU}))'),
            ("L19 = [<a,b,c,d,f,e>^1,<a,c,b,d,f,e>^1,<a,c,b,d,e>^1,<a,b,c,d,e>^1]",
             f'{CutType.SEQUENCE.value}(a, {CutType.PARALLEL.value}(b, c), d, {CutType.XOR.value}(f, {InductiveMiner.TAU}), e)')
        ]
    )
    def test_simple_event_logs(self, log: str, expected_tree: str):
        key, event_log = extract_traces_from_text(log)
        miner = InductiveMiner(event_log)
        miner.run()
        assert miner.process_tree_str == expected_tree
