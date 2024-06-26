from practical.ProcessMining.group2.conformance_checking.src.generate_footprint import (
    FootPrintMatrix,
)
from practical.ProcessMining.group2.conformance_checking.src.check_conformance import (
    ConformanceChecking,
)
from practical.ProcessMining.group2.conformance_checking.src.visualize_matrix import (
    visualize_sorted_dict,
)


from sortedcontainers import SortedDict, SortedSet


def test_cf():
    traces = SortedDict(
        {
            '1': ['a', 'b', 'c', 'd'],
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
            '2': ['a', 'b', 'c', 'd'],
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
            '3': ['a', 'b', 'c', 'd'],
            '30': ['a', 'd', 'c'],
            '31': ['b', 'c', 'd'],
            '32': ['b', 'c', 'd'],
            '33': ['b', 'c', 'd'],
            '34': ['b', 'c', 'd'],
            '35': ['b', 'c', 'd'],
            '4': ['a', 'b', 'c', 'd'],
            '5': ['a', 'b', 'c', 'd'],
            '6': ['a', 'b', 'c', 'd'],
            '7': ['a', 'b', 'c', 'd'],
            '8': ['a', 'b', 'c', 'd'],
            '9': ['a', 'b', 'c', 'd'],
        }
    )

    fp_matrix = FootPrintMatrix(traces)

    fp_matrix.get_transitions()

    fp_matrix.get_footprint_regular_alpha_miner()

    assert fp_matrix.relations == SortedDict(
        {
            'a': SortedDict({'a': '#', 'b': '->', 'c': '#', 'd': '||'}),
            'b': SortedDict({'a': '<-', 'b': '#', 'c': '->', 'd': '#'}),
            'c': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '->'}),
            'd': SortedDict({'a': '||', 'b': '#', 'c': '<-', 'd': '#'}),
        }
    )

    # visualize_sorted_dict(fp_matrix.relations, "")


def test_get_conformance_matrix():
    fpm_1 = FootPrintMatrix.from_relations(
        {
            'a': SortedDict({'a': '#', 'b': '->', 'c': '#', 'd': '||'}),
            'b': SortedDict({'a': '<-', 'b': '#', 'c': '->', 'd': '#'}),
            'c': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '->'}),
            'd': SortedDict({'a': '||', 'b': '#', 'c': '<-', 'd': '#'}),
        }
    )

    fpm_2 = FootPrintMatrix.from_relations(
        {
            'a': SortedDict({'a': '->', 'b': '->', 'c': '#', 'd': '||'}),
            'b': SortedDict({'a': '<-', 'b': '#', 'c': '->', 'd': '#'}),
            'c': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '->'}),
            'd': SortedDict({'a': '||', 'b': '->', 'c': '<-', 'd': '||'}),
        }
    )
    target = FootPrintMatrix.from_relations(
        {
            'a': SortedDict({'a': '#:->', 'b': '', 'c': '', 'd': ''}),
            'b': SortedDict({'a': '', 'b': '', 'c': '', 'd': ''}),
            'c': SortedDict({'a': '', 'b': '', 'c': '', 'd': ''}),
            'd': SortedDict({'a': '', 'b': '#:->', 'c': '', 'd': '#:||'}),
        }
    )

    cc = ConformanceChecking(fpm_1, fpm_2)
    assert cc.get_conformance_matrix().relations == target.relations
    visualize_sorted_dict(fpm_1.relations, "dict_1")
    # visualize_sorted_dict(fpm_2.relations, "dict_2")
    # visualize_sorted_dict(conformance_matrix.relations, "conf_check_1_2")


def test_get_conformance_value():
    fpm_1 = FootPrintMatrix.from_relations(
        {
            'a': SortedDict({'a': '#', 'b': '->', 'c': '#', 'd': '||'}),
            'b': SortedDict({'a': '<-', 'b': '#', 'c': '->', 'd': '#'}),
            'c': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '->'}),
            'd': SortedDict({'a': '||', 'b': '#', 'c': '<-', 'd': '#'}),
        }
    )

    fpm_2 = FootPrintMatrix.from_relations(
        {
            'a': SortedDict({'a': '->', 'b': '->', 'c': '#', 'd': '||'}),
            'b': SortedDict({'a': '<-', 'b': '#', 'c': '->', 'd': '#'}),
            'c': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '->'}),
            'd': SortedDict({'a': '||', 'b': '->', 'c': '<-', 'd': '||'}),
        }
    )
    # check / 0
    cc = ConformanceChecking(fpm_1, fpm_2)
    assert cc.get_conformance_value() == 0.8125
