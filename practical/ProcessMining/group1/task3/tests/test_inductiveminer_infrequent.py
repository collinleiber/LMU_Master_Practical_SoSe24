import pytest
from typing import List, Tuple, Set, Optional, Dict

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


    def test_get_frequent_directly_follows_graph(self):
        pass

    def test_calculate_eventually_follows_graph(self):
        pass

    def test_get_frequent_eventually_follows_graph(self):
        pass

