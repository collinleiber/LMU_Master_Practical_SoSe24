import copy
import os

import numpy as np
import pytest
import pm4py
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch

from practical.ProcessMining.group1.shared import utils
from practical.ProcessMining.group1.task2.alphaminer import AlphaMiner

dir_path = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = Path(os.path.join(dir_path, '../../shared/example_files'))

FILE_PATH_CSV = str(DATA_DIR / 'common-example.csv')  # event log from the paper
FILE_PATH_TXT = str(DATA_DIR / 'simple_event_logs.txt')
TEST_LOGS = utils.read_txt_test_logs(FILE_PATH_TXT)


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


@pytest.mark.parametrize(
    "key",
    [
        'dfg',
        'sequence',
        'parallel',
        'activities',
        'start_activities',
        'end_activities',
        'min_trace_length',
    ]
)
def test_footprints_discovery(alpha_miner: AlphaMiner, key: str, footprints: Dict[str, Any]) -> None:

    assert isinstance(footprints, dict), "The returned object is not a dictionary"

    # Check if the returned dictionary has the expected keys
    assert key in footprints, f"The dictionary does not contain the key {key}"

    # Check if discovered footprints match PM4PY (ignoring frequency information)
    assert alpha_miner.discover_footprints() == footprints, "Discovered footprints do not match PM4PY"


@pytest.mark.parametrize(
    "activity_id,expected_encoding",
    [
        (0, 'a'),
        (1, 'b'),
        (2, 'c'),
        (3, 'd'),
        (4, 'e'),
        (-1, KeyError),
    ]
)
def test_get_activity_name(alpha_miner: AlphaMiner, activity_id: int, expected_encoding: str) -> None:
    # Test decoding of activity ids
    if expected_encoding == KeyError:
        with pytest.raises(KeyError):
            alpha_miner._get_activity_name(activity_id), "Failed to raise exception for invalid activity id"
    else:
        assert alpha_miner._get_activity_name(activity_id) == expected_encoding, \
            "Activity name does not match expected value"


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


@pytest.mark.parametrize(
    "file",
    [
        'common-example.csv',
        'running-example.xes',
        'example.txt',
        'unknown_file.csv',
    ]
)
def test_import_event_log(file: str) -> None:
    log_path = str(DATA_DIR / file)
    if Path(log_path).exists():
        # Test valid file formats
        if file.endswith(".xes") or file.endswith('.csv'):
            valid_miner = AlphaMiner(log_path)
            assert isinstance(valid_miner, AlphaMiner), "Failed to import file"

        # Test unsupported file extension
        elif file.endswith(".txt"):
            with pytest.raises(Exception):
                AlphaMiner(log_path), "Failed to raise exception for unsupported file extension"

    else:
        # Test non-existent file
        with pytest.raises(Exception):
            AlphaMiner(log_path), "Failed to raise exception for non-existent file"


@pytest.mark.parametrize(
    "alpha_miner,expected_result",
    [
        (AlphaMiner(FILE_PATH_CSV),
         [({'a'}, {'b', 'e'}), ({'b', 'e'}, {'d'}), ({'c', 'e'}, {'d'}), ({'a'}, {'c', 'e'})]),
        (AlphaMiner(get_test_case(case="L1")),
         [({'a'}, {'e', 'b'}), ({'e', 'b'}, {'d'}), ({'e', 'c'}, {'d'}), ({'a'}, {'e', 'c'})]),
        (AlphaMiner(get_test_case(case="L2")),
         [({'f', 'a'}, {'c'}), ({'b'}, {'e', 'd'}), ({'f', 'a'}, {'b'}), ({'e'}, {'f'}), ({'c'}, {'e', 'd'})]),
        (AlphaMiner(get_test_case(case="L3")),
         [({'c'}, {'e'}), ({'b'}, {'c'}), ({'d'}, {'e'}), ({'e'}, {'f', 'g'}), ({'f', 'a'}, {'b'}), ({'b'}, {'d'})]),
        (AlphaMiner(get_test_case(case="L4")),
         [({'c'}, {'e', 'd'}), ({'a', 'b'}, {'c'})]),
        (AlphaMiner(get_test_case(case="L5")),
         [({'a', 'd'}, {'b'}), ({'a'}, {'e'}), ({'c'}, {'d'}), ({'e'}, {'f'}), ({'b'}, {'f', 'c'})]),
        (AlphaMiner(get_test_case(case="L6")),
         [({'a'}, {'e'}), ({'b'}, {'f'}), ({'c', 'd'}, {'g'}), ({'e', 'f'}, {'g'}), ({'e', 'd'}, {'g'}), ({'a'}, {'c'}), ({'f', 'c'}, {'g'}), ({'b'}, {'d'})]),
        (AlphaMiner(get_test_case(case="L7")),
         [({'a'}, {'c'})]),
        (AlphaMiner(get_test_case(case="L8")),
         [({'a'}, {'b'}), ({'b'}, {'d'})]),
        (AlphaMiner(get_test_case(case="L9")),
         [({'c'}, {'e', 'd'}), ({'a', 'b'}, {'c'})]),
        (AlphaMiner(get_test_case(case="L10")),
         []),
        (AlphaMiner(get_test_case(case="L11")),
         [({'a'}, {'b'}), ({'a'}, {'c'}), ({'b'}, {'c'})]),
        (AlphaMiner(get_test_case(case="L12")),
         [({'c'}, {'e', 'd'}), ({'a', 'b'}, {'c'})])
    ],
)
def test_get_maximal_pairs(alpha_miner: AlphaMiner, expected_result: list):

    maximal_pairs = alpha_miner.get_maximal_pairs()

    # Test if get_maximal_pairs returns the correct output
    assert all(item in expected_result for item in maximal_pairs)
    assert all(item in maximal_pairs for item in expected_result)

    # Check if maximal_pairs is a list
    assert isinstance(maximal_pairs, list), "maximal_pairs is not a list"

    # Check if all elements in maximal_pairs are tuples of sets
    assert all(
        isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], set) and isinstance(pair[1], set)
        for pair in maximal_pairs), "Not all elements in maximal_pairs are tuples of two sets"


@pytest.mark.parametrize(
    "miner,activity,expected_result",
    [
        (AlphaMiner(FILE_PATH_CSV), 0, [(0, (1, 4)), (0, (2, 4))]),
        (AlphaMiner(FILE_PATH_CSV), 1, [])
    ])
def test_right_side_maximization(miner: AlphaMiner, activity: int, expected_result: List):
    # Test if right_side_maximization returns the correct output
    result = miner._right_side_maximization(activity=activity)

    assert result == expected_result

    # Check if result is a list
    assert isinstance(result, list), "result is not a list"

    # Check if all elements in result are tuples
    assert all(isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], tuple)
               for pair in result), "Not all elements in result are tuples"


@pytest.mark.parametrize(
    "miner,activity,expected_result",
    [
        (AlphaMiner(FILE_PATH_CSV), 3, [((1, 4), 3), ((2, 4), 3)]),
        (AlphaMiner(FILE_PATH_CSV), 2, [])
    ])
def test_left_side_maximization(miner: AlphaMiner, activity: int, expected_result: List):
    # Test if left_side_maximization returns the correct output
    result = miner._left_side_maximization(activity=activity)

    assert result == expected_result

    # Check if result is a list
    assert isinstance(result, list), "result is not a list"

    # Check if all elements in result are tuples
    assert all(isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], tuple)
               for pair in result), "Not all elements in result are tuples"


@pytest.mark.parametrize(
    "miner,expected_result",
    [
        (AlphaMiner(FILE_PATH_CSV), []),
    ]
)
def test_prune_redundant_sequential_pairs(miner: AlphaMiner, expected_result: List):
    # Test if prune_redundant_sequential_pairs returns the correct output
    result = miner._prune_redundant_sequential_pairs()

    # Perform assertions based on expected output
    assert result == expected_result

    # Check if pruned_pairs is a list
    assert isinstance(result, list), "pruned_pairs is not a list"

    # Check if all elements in pruned_pairs are tuples
    assert all(isinstance(pair, tuple) for pair in result), "Not all elements in pruned_pairs are tuples"


@pytest.mark.parametrize(
    "is_encoded,field_name,sample_pairs,expected_result",
    [
        (False, None, [(1, 2), (3, 4), (0, (1, 2))],
         [(1, 2), (3, 4), (0, (1, 2))]),
        (True, None, [(1, 2), (3, 4), (0, (1, 2))],
         [({'b'}, {'c'}), ({'d'}, {'e'}), ({'a'}, {'b', 'c'})]),
        (True, 'sequential_pairs', None,
         [({'a'}, {'b'}), ({'a'}, {'e'}), ({'e'}, {'d'}), ({'c'}, {'d'}), ({'a'}, {'c'}), ({'b'}, {'d'})]),
        (True, 'parallel_pairs', None,
         [({'b'}, {'c'}), ({'c'}, {'b'})])
    ]
)
def test_activity_encoder(alpha_miner: AlphaMiner, is_encoded: bool, field_name, sample_pairs, expected_result):
    # Test if activity_encoder returns the correct output
    input_pairs = (sample_pairs if sample_pairs else getattr(alpha_miner, field_name))
    output_pairs = alpha_miner._activity_encoder(input_pairs, encoded=is_encoded, getter=True)

    # Check for different list, if encoding worked as expected
    assert expected_result == output_pairs

    if is_encoded:
        # Check if encoded_test is a list
        assert isinstance(output_pairs, list), "encoded_test is not a list"

        # Check if all elements in encoded_test are tuples of sets
        assert all(
            isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], set) and isinstance(pair[1], set)
            for pair in output_pairs), "Not all elements in encoded_test are tuples of two sets"


@pytest.mark.parametrize(
    "pair_type,field_name",
    [
        (">", "following_pairs"),
        ("||", "parallel_pairs"),
        ("->", "sequential_pairs"),
        ("#", "not_following_pairs"),
        ("<-", "before_pairs"),
        ("max", "maximal_pairs"),
    ]
)
def test_print_single_pair_type(alpha_miner: AlphaMiner, pair_type: str, field_name: np.ndarray or List):
    assert (np.array_equal(np.asarray(
        alpha_miner.print_single_pair_type(pair_type=pair_type, encoded=False, getter=True)),
        getattr(alpha_miner, field_name))
    ), "Returned list does not match to expected list"


def test_print_pairs(capfd, alpha_miner: AlphaMiner):
    alpha_miner.print_pairs(encoded=True)

    assert capfd.readouterr()


def test_visualization_calls(alpha_miner):
    with patch('practical.ProcessMining.group1.task2.alphaminer.pn_visualizer.apply') as mock_apply, \
            patch('practical.ProcessMining.group1.task2.alphaminer.pn_visualizer.view') as mock_view:
        alpha_miner.build_and_visualize_petrinet()

        mock_apply.assert_called_once()
        mock_view.assert_called_once()


def test_petrinet_structure(alpha_miner):
    alpha_miner.build_and_visualize_petrinet()

    net = alpha_miner.net
    initial_marking = alpha_miner.initial_marking
    final_marking = alpha_miner.final_marking

    # Check if the correct number of places and transitions are created
    assert len(net.places) > 0, "No places created in the Petri net."
    assert len(net.transitions) > 0, "No transitions created in the Petri net."

    # Verify initial and final markings are correctly assigned
    assert len(initial_marking) > 0, "Initial marking is not set."
    assert len(final_marking) > 0, "Final marking is not set."

    # check specific connections based on known input scenarios
    # require access to `transitions` and `places` by name or id


def test_transition_connections(alpha_miner):
    # check transitions are connected to the right place
    alpha_miner.build_and_visualize_petrinet()

    net = alpha_miner.net
    transitions = {t.label: t for t in net.transitions}
    places = net.places  # places is a set, so we'll iterate over it directly

    # Check connections from transitions to places
    global_start = None
    global_end = None
    for place in places:
        if place.name == 'global_start':
            global_start = place
        elif place.name == 'global_end':
            global_end = place

    # Verify global start and end are correctly found
    assert global_start, "Global start place not found in the Petri net."
    assert global_end, "Global end place not found in the Petri net."

    # Check connections from start place to initial transitions
    for activity in alpha_miner.t_in:
        activity_name = alpha_miner._get_activity_name(activity)
        assert global_start.out_arcs, "Global start place has no outgoing arcs."
        assert any(arc.target == transitions[activity_name] for arc in global_start.out_arcs), \
            f"Initial activity {activity_name} is not correctly connected from start place."

    # Check connections to end place from final transitions
    for activity in alpha_miner.t_out:
        activity_name = alpha_miner._get_activity_name(activity)
        assert global_end.in_arcs, "Global end place has no incoming arcs."
        assert any(arc.source == transitions[activity_name] for arc in global_end.in_arcs), \
            f"Terminal activity {activity_name} is not correctly connected to end place."


@pytest.mark.integration
def test_complete_petrinet_functionality(alpha_miner):
    # This test will incorporate visualization mock tests and structure tests
    with patch('practical.ProcessMining.group1.task2.alphaminer.pn_visualizer.apply') as mock_apply, \
         patch('practical.ProcessMining.group1.task2.alphaminer.pn_visualizer.view') as mock_view:

        test_visualization_calls(alpha_miner)
        test_petrinet_structure(alpha_miner)

        # Check the visualization methods were called as part of the process
        mock_apply.assert_called_once()
        mock_view.assert_called_once()
