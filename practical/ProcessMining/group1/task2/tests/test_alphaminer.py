import pytest
import pm4py
import pandas as pd
from practical.ProcessMining.group1.task2.alphaminer import AlphaMiner
from typing import Dict, Any

FILE_PATH = '../example_files/common-example.csv'  # event log from the paper


@pytest.fixture
def alpha_miner() -> AlphaMiner:
    return AlphaMiner(FILE_PATH)

@pytest.fixture
def event_log() -> pd.DataFrame:
    event_log = pd.read_csv(FILE_PATH, sep=';')
    return pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')


@pytest.fixture
def footprints(event_log: pd.DataFrame) -> Dict[str, Any]:
    footprints = pm4py.discovery.discover_footprints(event_log)
    # convert to set (discard frequency information of dictionary as our implementation does not provide that)
    footprints['dfg'] = set(footprints['dfg'].keys())
    return footprints


def test_footprints_discovery(alpha_miner: AlphaMiner, footprints: Dict[str, Any]) -> None:
    # test if discovered footprints match PM4PY (ignoring frequency information)
    assert alpha_miner.discover_footprints() == footprints


def test_get_maximal_pairs(alpha_miner):
    expected_result = [({'c'}, {'b'}), ({'a'}, {'b', 'e'}), ({'c', 'e'}, {'d'}), ({'b', 'e'}, {'d'}), ({'a'}, {'c', 'e'})]

    maximal_pairs = alpha_miner.get_maximal_pairs()

    # Test if get_maximal_pairs returns the correct output
    assert expected_result == maximal_pairs

    # Check if maximal_pairs is a list
    assert isinstance(maximal_pairs, list), "maximal_pairs is not a list"

    # Check if all elements in maximal_pairs are tuples of sets
    assert all(
        isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], set) and isinstance(pair[1], set)
        for pair in maximal_pairs), "Not all elements in maximal_pairs are tuples of two sets"


def test_right_side_maximization(alpha_miner):
    # Test if right_side_maximization returns the correct output
    expected_result_0 = [(0, (1, 4)), (0, (2, 4))]
    expected_result_1 = []

    right_side_0 = alpha_miner._right_side_maximization(activity=0)
    right_side_1 = alpha_miner._right_side_maximization(activity=1)

    assert right_side_0 == expected_result_0
    assert right_side_1 == expected_result_1

    # Check if right_side_0 is a list
    assert isinstance(right_side_0, list), "right_side_0 is not a list"

    # Check if all elements in right_side_0 are tuples
    assert all(isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], tuple)
               for pair in right_side_0), "Not all elements in right_side_0 are tuples"


def test_left_side_maximization(alpha_miner):
    expected_result_3 = [((1, 4), 3), ((2, 4), 3)]
    expected_result_2 = []

    left_side_3 = alpha_miner._left_side_maximization(activity=3)
    left_side_2 = alpha_miner._left_side_maximization(activity=2)

    assert left_side_3 == expected_result_3
    assert left_side_2 == expected_result_2

    # Check if left_side_3 is a list
    assert isinstance(left_side_3, list), "left_side_3 is not a list"

    # Check if all elements in left_side_3 are tuples
    assert all(isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], tuple)
               for pair in left_side_3), "Not all elements in left_side_3 are tuples"


def test_prune_redundant_sequential_pairs(alpha_miner: AlphaMiner):
    # Test if prune_redundant_sequential_pairs returns the correct output
    split_result = [(0, (1, 4)), (0, (2, 4))]  # test_right_side_maximization(alpha_miner)
    join_result = [((1, 4), 3), ((2, 4), 3)]  # test_left_side_maximization(alpha_miner)
    expected_result = [(2, 1)]

    pruned_pairs = alpha_miner._prune_redundant_sequential_pairs(split_result, join_result)

    # Check if pruned_pairs is a list
    assert isinstance(pruned_pairs, list), "pruned_pairs is not a list"

    # Check if all elements in pruned_pairs are tuples
    assert all(isinstance(pair, tuple) for pair in pruned_pairs), "Not all elements in pruned_pairs are tuples"

    # Perform assertions based on expected output
    assert pruned_pairs == expected_result


def test_activity_encoder(alpha_miner: AlphaMiner):
    # Test if activity_encoder returns the correct output
    test_pairs = [(1, 2), (3, 4), (0, (1, 2))]  # Example input pairs

    expected_test = [({'b'}, {'c'}), ({'d'}, {'e'}), ({'a'}, {'b', 'c'})]
    expected_sequential = [({'a'}, {'b'}), ({'a'}, {'e'}), ({'c'}, {'b'}), ({'e'}, {'d'}), ({'c'}, {'d'}), ({'a'}, {'c'}), ({'b'}, {'d'})]
    expected_parallels = [({'b'}, {'c'})]

    decoded_test = alpha_miner._activity_encoder(test_pairs, encoded=False, getter=True)
    encoded_test = alpha_miner._activity_encoder(test_pairs, encoded=True, getter=True)
    encoded_parallels = alpha_miner._activity_encoder(alpha_miner.parallel_pairs, encoded=True, getter=True)
    encoded_sequential = alpha_miner._activity_encoder(alpha_miner.sequential_pairs, encoded=True, getter=True)

    # Check for different list, if encoding worked
    assert decoded_test == test_pairs
    assert encoded_test == expected_test
    assert encoded_parallels == expected_parallels
    assert encoded_sequential == expected_sequential

    # Check if encoded_test is a list
    assert isinstance(encoded_test, list), "encoded_test is not a list"

    # Check if all elements in encoded_test are tuples of sets
    assert all(
        isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], set) and isinstance(pair[1], set)
        for pair in encoded_test), "Not all elements in encoded_test are tuples of two sets"
