import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, ".."))

from pm4py.objects.log.obj import Trace, Event, EventLog


from src.generate_footprint import (
    FootPrintMatrix,
)
from src.check_conformance import (
    ConformanceChecking,
)
from src.visualize_matrix import (
    visualize_sorted_dict,
)
from src.comparison import Comparison, AlgoPm4Py


def test_cf():
    trace_data = {
        '1': ['a', 'b', 'c', 'd'],
        '2': ['a', 'b', 'c', 'd'],
        '3': ['a', 'b', 'c', 'd'],
        '4': ['a', 'b', 'c', 'd'],
        '5': ['a', 'b', 'c', 'd'],
        '6': ['a', 'b', 'c', 'd'],
        '7': ['a', 'b', 'c', 'd'],
        '8': ['a', 'b', 'c', 'd'],
        '9': ['a', 'b', 'c', 'd'],
        '10': ['a', 'b', 'c', 'd'],
        '11': ['d', 'a', 'b'],
        '12': ['d', 'a', 'b'],
        '13': ['d', 'a', 'b'],
        '14': ['d', 'a', 'b'],
        '15': ['d', 'a', 'b'],
        '16': ['d', 'a', 'b'],
        '17': ['d', 'a', 'b'],
        '18': ['d', 'a', 'b'],
        '19': ['d', 'a', 'b'],
        '20': ['d', 'a', 'b'],
        '21': ['a', 'd', 'c'],
        '22': ['a', 'd', 'c'],
        '23': ['a', 'd', 'c'],
        '24': ['a', 'd', 'c'],
        '25': ['a', 'd', 'c'],
        '26': ['a', 'd', 'c'],
        '27': ['a', 'd', 'c'],
        '28': ['a', 'd', 'c'],
        '29': ['a', 'd', 'c'],
        '30': ['a', 'd', 'c'],
        '31': ['b', 'c', 'd'],
        '32': ['b', 'c', 'd'],
        '33': ['b', 'c', 'd'],
        '34': ['b', 'c', 'd'],
        '35': ['b', 'c', 'd'],
    }
    traces = EventLog()

    for trace_id, activities in trace_data.items():
        trace = Trace()
        trace.attributes['concept:name'] = trace_id
        for activity in activities:
            event = Event({'concept:name': activity})
            trace.append(event)
        traces.append(trace)

    fp_matrix = FootPrintMatrix(traces)

    fp_matrix.generate_footprint()
    assert fp_matrix.relations == {
        'a': {'a': '#', 'b': '->', 'c': '#', 'd': '||'},
        'b': {'a': '<-', 'b': '#', 'c': '->', 'd': '#'},
        'c': {'a': '#', 'b': '<-', 'c': '#', 'd': '||'},
        'd': {'a': '||', 'b': '#', 'c': '||', 'd': '#'},
    }
    # visualize_sorted_dict(fp_matrix.relations, "")


def test_get_conformance_matrix():
    fpm_1 = FootPrintMatrix.from_relations(
        {
            'a': {'a': '#', 'b': '->', 'c': '#', 'd': '||'},
            'b': {'a': '<-', 'b': '#', 'c': '->', 'd': '#'},
            'c': {'a': '#', 'b': '<-', 'c': '#', 'd': '->'},
            'd': {'a': '||', 'b': '#', 'c': '<-', 'd': '#'},
        }
    )

    fpm_2 = FootPrintMatrix.from_relations(
        {
            'a': {'a': '->', 'b': '->', 'c': '#', 'd': '||'},
            'b': {'a': '<-', 'b': '#', 'c': '->', 'd': '#'},
            'c': {'a': '#', 'b': '<-', 'c': '#', 'd': '->'},
            'd': {'a': '||', 'b': '->', 'c': '<-', 'd': '||'},
        }
    )
    target = FootPrintMatrix.from_relations(
        {
            'a': {'a': '#:->', 'b': '', 'c': '', 'd': ''},
            'b': {'a': '', 'b': '', 'c': '', 'd': ''},
            'c': {'a': '', 'b': '', 'c': '', 'd': ''},
            'd': {'a': '', 'b': '#:->', 'c': '', 'd': '#:||'},
        }
    )

    cc = ConformanceChecking()
    assert cc.get_conformance_matrix(fpm_1, fpm_2).relations == target.relations
    visualize_sorted_dict(fpm_1.relations, "dict_1")


def test_get_conformance_value():
    fpm_1 = FootPrintMatrix.from_relations(
        {
            'a': {'a': '#', 'b': '->', 'c': '#', 'd': '||'},
            'b': {'a': '<-', 'b': '#', 'c': '->', 'd': '#'},
            'c': {'a': '#', 'b': '<-', 'c': '#', 'd': '->'},
            'd': {'a': '||', 'b': '#', 'c': '<-', 'd': '#'},
        }
    )

    fpm_2 = FootPrintMatrix.from_relations(
        {
            'a': {'a': '->', 'b': '->', 'c': '#', 'd': '||'},
            'b': {'a': '<-', 'b': '#', 'c': '->', 'd': '#'},
            'c': {'a': '#', 'b': '<-', 'c': '#', 'd': '->'},
            'd': {'a': '||', 'b': '->', 'c': '<-', 'd': '||'},
        }
    )
    # check / 0
    cc = ConformanceChecking()
    assert cc.get_conformance_value(fpm_1, fpm_2) == 0.8125


def test_pipeline():
    # TODO
    pass


def test_log_2_log():
    log_path = "InputLogs/L1.csv"
    comparison = Comparison()
    result = comparison.log_2_log(log_path, AlgoPm4Py.ALPHA)
    assert len(result) == 4
    for sublog, comparison_value in result:
        assert 0 <= comparison_value <= 1


def test_log_2_model():
    log_path = "InputLogs/L1.csv"
    comparison = Comparison()
    result = comparison.log_2_model(log_path, AlgoPm4Py.ALPHA)
    assert result == 1.0


def test_model_2_model_scenario_1():
    log_path = "InputLogs/L1.csv"
    comparison = Comparison()
    assert comparison.model_2_model(
        log_path,
        [
            AlgoPm4Py.ALPHA,
            AlgoPm4Py.ALPHAPLUS,
            AlgoPm4Py.HEURISTICMINER,
            AlgoPm4Py.INDUCTIVEMINER,
        ],
        1,
    ) == {
        'InputLog vs. AlgoPm4Py.ALPHA': 1.0,
        'InputLog vs. AlgoPm4Py.ALPHAPLUS': 1.0,
        'InputLog vs. AlgoPm4Py.HEURISTICMINER': 1.0,
        'InputLog vs. AlgoPm4Py.INDUCTIVEMINER': 1.0,
    }


def test_model_2_model_scenario_2():
    log_path = "InputLogs/L1.csv"
    comparison = Comparison()
    assert comparison.model_2_model(
        log_path,
        [
            AlgoPm4Py.ALPHA,
            AlgoPm4Py.ALPHAPLUS,
            AlgoPm4Py.HEURISTICMINER,
            AlgoPm4Py.INDUCTIVEMINER,
        ],
        2,
    ) == {
        'AlgoPm4Py.ALPHA vs. AlgoPm4Py.ALPHAPLUS': 1.0,
        'AlgoPm4Py.ALPHA vs. AlgoPm4Py.HEURISTICMINER': 1.0,
        'AlgoPm4Py.ALPHA vs. AlgoPm4Py.INDUCTIVEMINER': 1.0,
        'AlgoPm4Py.ALPHAPLUS vs. AlgoPm4Py.HEURISTICMINER': 1.0,
        'AlgoPm4Py.ALPHAPLUS vs. AlgoPm4Py.INDUCTIVEMINER': 1.0,
        'AlgoPm4Py.HEURISTICMINER vs. AlgoPm4Py.INDUCTIVEMINER': 1.0,
    }
