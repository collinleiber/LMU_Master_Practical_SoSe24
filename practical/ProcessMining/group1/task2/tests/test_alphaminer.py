import copy
import numpy as np
import pytest
import pm4py
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from practical.ProcessMining.group1.shared import utils
from practical.ProcessMining.group1.task2.alphaminer import AlphaMiner

FILE_PATH_CSV = '../example_files/common-example.csv'  # event log from the paper
TEST_LOGS = utils.read_txt_test_logs('../example_files/simple_event_logs.txt')


def get_test_case(case: str):
    cleaned = utils.deduplicate_list(TEST_LOGS[case])
    return utils.event_log_to_csv(cleaned)


@pytest.fixture
def alpha_miner() -> AlphaMiner:
    return AlphaMiner(FILE_PATH_CSV)


@pytest.fixture
def event_log() -> pd.DataFrame:
    event_log = pd.read_csv(FILE_PATH_CSV, sep=';')
    return pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')


@pytest.fixture
def footprints(event_log: pd.DataFrame) -> Dict[str, Any]:
    footprints = pm4py.discovery.discover_footprints(event_log)
    # convert to set (discard frequency information of dictionary as our implementation does not provide that)
    footprints['dfg'] = set(footprints['dfg'].keys())
    return footprints


def test_footprints_discovery(alpha_miner: AlphaMiner, footprints: Dict[str, Any]) -> None:
    # Check if the returned dictionary has the expected keys
    assert isinstance(footprints, dict), "The returned object is not a dictionary"
    assert 'dfg' in footprints, "The dictionary does not contain the key 'dfg'"
    assert 'sequence' in footprints, "The dictionary does not contain the key 'sequence'"
    assert 'parallel' in footprints, "The dictionary does not contain the key 'parallel'"
    assert 'activities' in footprints, "The dictionary does not contain the key 'activities'"
    assert 'start_activities' in footprints, "The dictionary does not contain the key 'start_activities'"
    assert 'end_activities' in footprints, "The dictionary does not contain the key 'end_activities'"
    assert 'min_trace_length' in footprints, "The dictionary does not contain the key 'min_trace_length'"

    # Check if discovered footprints match PM4PY (ignoring frequency information)
    assert alpha_miner.discover_footprints() == footprints, "Discovered footprints do not match PM4PY"


def test_get_activity_name(alpha_miner: AlphaMiner) -> None:
    # Test decoding of activity ids
    assert alpha_miner._get_activity_name(0) == 'a', "Activity name does not match expected value"
    assert alpha_miner._get_activity_name(1) == 'b', "Activity name does not match expected value"
    assert alpha_miner._get_activity_name(2) == 'c', "Activity name does not match expected value"
    assert alpha_miner._get_activity_name(3) == 'd', "Activity name does not match expected value"
    assert alpha_miner._get_activity_name(4) == 'e', "Activity name does not match expected value"
    with pytest.raises(KeyError):
        alpha_miner._get_activity_name(-1), "Failed to raise exception for invalid activity id"


def test_footprint_matrix(alpha_miner: AlphaMiner) -> None:
    miner = copy.deepcopy(alpha_miner)
    miner.parallel_pairs = miner.parallel_pairs[:1]
    matrix = miner.footprint_matrix()

    assert isinstance(matrix, pd.DataFrame), "The returned object is not a DataFrame"
    assert matrix.shape == (len(miner.activities), len(miner.activities)), "The DataFrame has incorrect shape"

    # Check if the DataFrame contains the expected values
    for pair in alpha_miner.all_pairs:
        a1, a2 = pair
        a1_value, a2_value = miner.activities[a1], miner.activities[a2]
        if any(np.array_equal(pair, p) for p in miner.parallel_pairs):
            assert matrix.at[a1_value, a2_value] == '||'
        elif any(np.array_equal(pair, p) for p in miner.sequential_pairs):
            assert matrix.at[a1_value, a2_value] == '→'
        elif any(np.array_equal(pair, p) for p in miner.not_following_pairs):
            assert matrix.at[a1_value, a2_value] == '#'
        elif any(np.array_equal(pair, p) for p in miner.before_pairs):
            assert matrix.at[a1_value, a2_value] == '←'
        else:
            assert matrix.at[a1_value, a2_value] == ''


def test_import_event_log(alpha_miner: AlphaMiner, tmp_path) -> None:
    # Test xes file
    xes_miner = AlphaMiner(FILE_PATH_XES)
    assert isinstance(xes_miner, AlphaMiner), "Failed to import XES file"

    # Test unsupported file extension
    tmp_txt = tmp_path / FILE_PATH_TXT
    with pytest.raises(Exception):
        txt_miner = AlphaMiner(FILE_PATH_TXT), "Failed to raise exception for unsupported file extension"

    # Test non-existent file
    with pytest.raises(Exception):
        unknown_miner = AlphaMiner("example_files/unknown_file.csv"), "Failed to raise exception for non-existent file"


def test_get_maximal_pairs(alpha_miner):
    expected_result = [({'a'}, {'b', 'e'}), ({'b', 'e'}, {'d'}), ({'c', 'e'}, {'d'}), ({'a'}, {'c', 'e'})]

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
    expected_result = []  # todo test alpha miner1, since pruning result not []

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
    expected_sequential = [({'a'}, {'b'}), ({'a'}, {'e'}), ({'e'}, {'d'}), ({'c'}, {'d'}),
                           ({'a'}, {'c'}), ({'b'}, {'d'})]
    expected_parallels = [({'b'}, {'c'}), ({'c'}, {'b'})]

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


def test_print_single_pair_type(alpha_miner: AlphaMiner):
    assert (np.array_equal(np.asarray(alpha_miner.print_single_pair_type(">", encoded=False, getter=True)),
                           alpha_miner.following_pairs)
            ), "Function is incorrect printing following pairs"
    assert (np.array_equal(np.asarray(alpha_miner.print_single_pair_type("||", encoded=False, getter=True)),
                           alpha_miner.parallel_pairs)
            ), "Function is incorrect printing parallel pairs"
    assert (np.array_equal(np.asarray(alpha_miner.print_single_pair_type("->", encoded=False, getter=True)),
                           alpha_miner.sequential_pairs)
            ), "Function is incorrect printing sequential pairs"
    assert (np.array_equal(np.asarray(alpha_miner.print_single_pair_type("#", encoded=False, getter=True)),
                           alpha_miner.not_following_pairs)
            ), "Function is incorrect printing not following pairs"
    assert (np.array_equal(np.asarray(alpha_miner.print_single_pair_type("<-", encoded=False, getter=True)),
                           alpha_miner.before_pairs)
            ), "Function is incorrect printing before pairs"
    assert (np.array_equal(np.asarray(alpha_miner.print_single_pair_type("max", encoded=False, getter=True)),
                           alpha_miner.maximal_pairs)
            ), "Function is incorrect printing maximal pairs"
