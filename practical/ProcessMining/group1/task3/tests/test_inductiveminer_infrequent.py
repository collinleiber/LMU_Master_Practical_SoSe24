from practical.ProcessMining.group1.shared.utils import check_lists_of_sets_equal
from practical.ProcessMining.group1.task3.inductiveminer import CutType
from practical.ProcessMining.group1.task3.inductiveminer_infrequent import InductiveMinerInfrequent


class TestInductiveMinerInfrequent:
    def test_handle_base_case_filtered(self):
        # basic logic
        log = [('a',), ('a',), ('a',)]
        miner = InductiveMinerInfrequent(log)
        base_cut, operator = miner._handle_base_cases_filtered(log)
        assert base_cut == [set('a')]
        assert operator == CutType.NONE
        # never or once
        log = [('',), ('a',), ('a',)]
        miner = InductiveMinerInfrequent(log)
        base_cut, operator = miner._handle_base_cases_filtered(log)
        assert check_lists_of_sets_equal(base_cut, [set('a'), set('𝜏')])  # order does not matter
        assert operator == CutType.XOR
        # once or more than once
        log = [('a',), ('a', 'a'), ('a', 'a', 'a')]
        miner = InductiveMinerInfrequent(log)
        base_cut, operator = miner._handle_base_cases_filtered(log)
        assert base_cut == [set('a'), set('𝜏')]  # order does matter
        assert operator == CutType.LOOP
        # never, once or more than once
        log = [('',), ('a',), ('a', 'a')]
        miner = InductiveMinerInfrequent(log)
        base_cut, operator = miner._handle_base_cases_filtered(log)
        assert base_cut == [set('𝜏'), set('a')]  # order does matter
        assert operator == CutType.LOOP

        # infrequent logic
        # single activity filtering
        log = [('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',),
               ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',), ('a',),
               ('a', 'a', 'a'), ('',)]
        miner = InductiveMinerInfrequent(log, threshold=0.5)
        base_cut, operator = miner._handle_base_cases_filtered(log)
        assert base_cut == [set('a')]
        assert operator == CutType.NONE
        # empty trace filtering
        log = [('a',), ('a',), ('a',),
               ('a', 'a', 'a'), ('a', 'a', 'a'), ('a', 'a', 'a'), ('a', 'a', 'a'), ('a', 'a', 'a'),
               ('',), ('',)]
        miner = InductiveMinerInfrequent(log, threshold=0.25)
        base_cut, operator = miner._handle_base_cases_filtered(log)
        assert base_cut == [set('a'), set('𝜏')]  # order does matter
        assert operator == CutType.LOOP

