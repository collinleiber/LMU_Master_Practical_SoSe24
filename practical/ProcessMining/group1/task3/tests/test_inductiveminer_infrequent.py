import pytest
from typing import List, Tuple, Set, Optional, Dict
from unittest.mock import patch
from practical.ProcessMining.group1.shared.utils import check_lists_of_sets_equal
from practical.ProcessMining.group1.task3.inductiveminer import CutType
from practical.ProcessMining.group1.task3.inductiveminer_infrequent import InductiveMinerInfrequent


class TestInductiveMinerInfrequent:
    @pytest.fixture
    def dummy_miner(self):
        return InductiveMinerInfrequent([('a', 'c'), ('b')], 0.5)

    @pytest.mark.parametrize(
        "log,threshold,expected_cut,expected_operator",
        [
            ([('a',), ('a',), ('a',)], 0.0,  # basic logic
             [set('a')], CutType.NONE),
            ([('',), ('a',), ('a',)], 0.0,  # never or once
             [set('a'), set('𝜏')], CutType.XOR),
            ([('a',), ('a', 'a'), ('a', 'a', 'a')], 0.0,  # once or more than once
             [set('a'), set('𝜏')], CutType.LOOP),
            ([('',), ('a',), ('a', 'a')], 0.0,  # never, once or more than once
             [set('𝜏'), set('a')], CutType.LOOP),
            ([('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',),
              ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',),
              ('a', 'a', 'a'), ('',)], 0.5,
             [set('a')], CutType.NONE),  # single activity filtering
            ([('a',), ('a',), ('a',),
              ('a', 'a', 'a'), ('a', 'a', 'a'), ('a', 'a', 'a'), ('a', 'a', 'a'), ('a', 'a', 'a'),
              ('',), ('',)], 0.25,
             [set('a'), set('𝜏')], CutType.LOOP),  # empty trace filtering
        ]
    )
    def test_handle_base_case_filtered(self, log: List[Tuple[str]], threshold: Optional[float],
                                       expected_cut: List[Set], expected_operator: CutType):
        miner = InductiveMinerInfrequent(event_log=log, threshold=threshold)
        base_cut, operator = miner._handle_base_cases_filtered(log)
        assert base_cut == expected_cut
        assert operator == expected_operator

    @pytest.mark.parametrize(
        "base_dfg,frequent_dfg",
        [
            ({('a', 'c'): 1, ('c', 'd'): 1, ('d', 'e'): 1, ('e', 'b'): 1, ('a', 'b'): 1, ('b', 'a'): 1, ('a', 'e'): 2,
              ('e', 'd'): 1, ('d', 'c'): 1, ('e', 'c'): 1, ('c', 'b'): 1, ('b', 'd'): 1, ('a', 'd'): 1, ('d', 'b'): 1,
              ('b', 'c'): 1, ('c', 'e'): 1},
             {('a', 'b'): 1, ('a', 'c'): 1, ('a', 'd'): 1, ('a', 'e'): 2, ('b', 'a'): 1, ('b', 'c'): 1, ('b', 'd'): 1,
              ('c', 'b'): 1, ('c', 'd'): 1, ('c', 'e'): 1, ('d', 'b'): 1, ('d', 'c'): 1, ('d', 'e'): 1, ('e', 'b'): 1,
              ('e', 'c'): 1, ('e', 'd'): 1}),

        ]
    )
    def test_get_frequent_directly_follows_graph(self, dummy_miner: InductiveMinerInfrequent,
                                                 base_dfg: Dict[Tuple[str, str], int],
                                                 frequent_dfg: Dict[Tuple[str, str], int]):
        result_dfg = dummy_miner.get_frequent_directly_follows_graph(base_dfg)
        assert result_dfg == frequent_dfg

    @pytest.mark.parametrize(
        "log,expected_efg,expected_fefg",
        [
            ([('a', 'c', 'd', 'e', 'b'),
              ('a', 'b', 'a', 'e', 'd', 'c'),
              ('a', 'e', 'c', 'b', 'd'),
              ('a', 'd', 'b', 'c', 'e')],
             {('a', 'c'): 5, ('a', 'd'): 5, ('a', 'e'): 5, ('a', 'b'): 4, ('c', 'd'): 2, ('c', 'e'): 2, ('c', 'b'): 2,
              ('d', 'e'): 2, ('d', 'b'): 2, ('e', 'b'): 2, ('a', 'a'): 1, ('b', 'a'): 1, ('b', 'e'): 2, ('b', 'd'): 2,
              ('b', 'c'): 2, ('e', 'd'): 2, ('e', 'c'): 2, ('d', 'c'): 2},
             {('a', 'b'): 4, ('a', 'c'): 5, ('a', 'd'): 5, ('a', 'e'): 5, ('b', 'a'): 1, ('b', 'c'): 2, ('b', 'd'): 2,
              ('b', 'e'): 2, ('c', 'b'): 2, ('c', 'd'): 2, ('c', 'e'): 2, ('d', 'b'): 2, ('d', 'c'): 2, ('d', 'e'): 2,
              ('e', 'b'): 2, ('e', 'c'): 2, ('e', 'd'): 2}),

            ([('a', 'b', 'c', 'a', 'b', 'd', 'a', 'b', 'c', 'a', 'b', 'd'),
              ('c', 'a', 'b', 'a', 'b', 'c', 'a', 'b'),
              ('a', 'd', 'a', 'b', 'a', 'g')],
             {('a', 'b'): 18, ('a', 'c'): 6, ('a', 'a'): 12, ('a', 'd'): 7, ('b', 'c'): 6, ('b', 'a'): 10,
              ('b', 'b'): 9, ('b', 'd'): 6, ('c', 'a'): 8, ('c', 'b'): 8, ('c', 'd'): 3, ('c', 'c'): 2, ('d', 'a'): 4,
              ('d', 'b'): 3, ('d', 'c'): 1, ('d', 'd'): 1, ('a', 'g'): 3, ('d', 'g'): 1, ('b', 'g'): 1},
             {('a', 'a'): 12, ('a', 'b'): 18, ('a', 'c'): 6, ('a', 'd'): 7, ('b', 'a'): 10, ('b', 'b'): 9,
              ('b', 'c'): 6, ('b', 'd'): 6, ('c', 'a'): 8, ('c', 'b'): 8, ('c', 'd'): 3, ('d', 'a'): 4, ('d', 'b'): 3})
        ]
    )
    def test_get_frequent_eventually_follows_graph(self, log: List[Tuple[str]], expected_efg: Dict[Tuple[str, str], int],
                                                   expected_fefg: Dict[Tuple[str, str], int]):
        miner = InductiveMinerInfrequent(log, 0.3)

        result_efg = miner._calculate_eventually_follows_graph(log)
        assert result_efg == expected_efg

        freq_efg = miner.get_frequent_eventually_follows_graph(log)
        assert freq_efg == expected_fefg


    @pytest.mark.parametrize(
        "log,given_cut,expected_filter",
        [
            ([("A", "A"), ("A", "A", "A", "A", "B", "A", "A", "A"), ("B", "B", "B", "C")],
             [{"A"}, {"B", "C"}],
             [[('A', 'A'), ('A', 'A', 'A', 'A', 'A', 'A', 'A')], [('B', 'B', 'B', 'C')]]),

            ([("A", "A"), ("A", "A", "A", "A", "B", "A", "A", "A"), ("B", "B", "B")],
             [{"A"}, {"B"}],
             [[('A', 'A'), ('A', 'A', 'A', 'A', 'A', 'A', 'A')], [('B', 'B', 'B')]]),

            ([("A", "A")],
             [{"A"}, {"B"}],
             [[('A', 'A')]]),

            ([("a", "b", "c"), ("d", "e", "f", "g"), ("a", "b"), ("d", "e")],
             [{"a", "b", "c", "g"}, {"d", "e", "f"}],
             [[("a", "b", "c"), ("a", "b")], [("d", "e", "f"), ("d", "e")]])
        ]
    )
    def test_xor_split_filtered(self, log: List[Tuple[str]], given_cut: List[Set[str]],
                                expected_filter: List[Tuple[str]]):
        miner = InductiveMinerInfrequent(event_log=log, threshold=0.0)

        result = miner.xor_split_filtered(log=log, groups=given_cut)
        assert result == expected_filter

    @pytest.mark.parametrize(
        "log,given_cut,expected_filter",
        [
            ([("A", "A", "A", "A", "B", "B", "B", "C", "A", "C")],
             [{"A"}, {"B"}, {"C"}],
             [[('A', 'A', 'A', 'A')], [('B', 'B', 'B')], [('C', 'C')]]),

            ([("A", "A", "A", "A", "B", "B", "B", "C", "C", "A", "C", "A", "C")],
             [{"A"}, {"B"}, {"C"}],
             [[('A', 'A', 'A', 'A')], [('B', 'B', 'B')], [('C', 'C', 'C', 'C')]]),

            ([("A", "A", "A", "A", "B", "B", "B", "C", "B", "C", "B", "C", "C", "D")],
             [{"A"}, {"B", "C"}, {"D"}],
             [[("A", "A", "A", "A")], [("B", "B", "B", "C", "B", "C", "B", "C", "C")], [("D", )]]),

            ([("A", "A", "A", "C", "B", "B", "A", "C", "B", "C", "C", "D", "A", "D", "B")],
             [{"A"}, {"B", "C"}, {"D"}],
             [[("A", "A", "A")], [("C", "B", "B", "C", "B", "C", "C")], [("D", "D")]])
        ]
    )
    def test_sequence_split_filtered(self, log: List[Tuple[str]], given_cut: List[Set[str]],
                                     expected_filter: List[Tuple[str]]):
        miner = InductiveMinerInfrequent(event_log=log, threshold=0.4)

        result = miner.sequence_split_filtered(log=log, groups=given_cut)
        assert result == expected_filter

    @pytest.mark.parametrize(
        "log,given_cut,expected_filter",
        [
            ([("B", "A", "B"), ("C", "A", "C")],
             [{"A"}, {"B"}, {"C"}],
             [sorted([('A',), ('',), ('',), ('A',), ('',), ('',)]),
              sorted([('B',), ('B',)]),
              sorted([('C',), ('C',)])]),

            ([("D", "A", "B", "D"), ("C", "B", "A", "C")],
             [{"A", "B"}, {"C"}, {"D"}],
             [sorted([('A', 'B'), ('',), ('',), ('B', 'A'), ('',), ('',)]),
              sorted([('C',), ('C',)]),
              sorted([('D',), ('D',)])]),

            ([("B", "A", "B")],
             [{"A"}, {"B"}],
             [sorted([('A',), ('',), ('',)]),
              sorted([('B',), ('B',)])]),

            ([("B", "A", "B")],
             [{"B"}, {"A"}],
             [sorted([('B',), ('B',)]),
              sorted([('A',)])]),
        ]
    )
    def test_loop_split_filtered(self, log: List[Tuple[str]], given_cut: List[Set[str]],
                                 expected_filter: List[Tuple[str]]):
        assert len(given_cut) >= 2, "Invalid cut, loop split requires at least two groups"
        miner = InductiveMinerInfrequent(event_log=log, threshold=0.4)

        result = miner.loop_split_filtered(log=log, groups=given_cut)
        result[0] = sorted(result[0])
        assert result == expected_filter


    @pytest.mark.parametrize(
        "log,threshold",
        [
            ([('a', 'c', 'e', 'g'),('a', 'e', 'c', 'g'),('b', 'd', 'f', 'g'),('b', 'f', 'd', 'g')], 0.1)
        ]
    )
    @patch('practical.ProcessMining.group1.task3.inductiveminer.pt_visualizer.view')
    @patch('practical.ProcessMining.group1.task3.inductiveminer.pt_visualizer.save')
    def test_visualize_process_tree(self, mock_save, mock_view, log: List[Tuple[str]], threshold: float):
        miner = InductiveMinerInfrequent(log, threshold)
        miner.run()

        miner.visualize_process_tree()

        # Check if the view method was called
        mock_view.assert_called_once()

        # optionally check if the save method was called when actually saving the image
        # mock_save.assert_called_once_with(gviz, "process_tree.png")

