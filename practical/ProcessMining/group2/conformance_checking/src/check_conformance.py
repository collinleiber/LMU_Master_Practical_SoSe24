import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from src.generate_footprint import (
    FootPrintMatrix,
)


class ConformanceChecking:
    # Initialize ConformanceChecking Object

    # Checks two dictionaries (footprint)
    # TODO only works if keys of dicts are the same
    def get_conformance_matrix(self, fpm_1, fpm_2):
        dict_out = {}
        for (outer_k1, outer_v1), (outer_k2, outer_v2) in zip(
            fpm_1.relations.items(), fpm_2.relations.items()
        ):
            inner_dict_out = {}
            for (inner_k1, inner_v1), (inner_k2, inner_v2) in zip(
                outer_v1.items(), outer_v2.items()
            ):
                if inner_v1 == inner_v2:
                    inner_dict_out[inner_k1] = ''
                else:
                    inner_dict_out[inner_k1] = '{}:{}'.format(inner_v1, inner_v2)

            dict_out[outer_k1] = inner_dict_out

        return FootPrintMatrix.from_relations(dict_out)

    def get_conformance_value(self, fpm_1, fpm_2):
        print("Calculating Conformance Value!")
        different_cells = 0
        total_cells = len(fpm_1.relations) ** 2
        for (outer_k1, outer_v1), (outer_k2, outer_v2) in zip(
            fpm_1.relations.items(), fpm_2.relations.items()
        ):
            for (inner_k1, inner_v1), (inner_k2, inner_v2) in zip(
                outer_v1.items(), outer_v2.items()
            ):
                if inner_v1 != inner_v2:
                    # print("diff at: ", outer_k1, inner_k1)
                    # print(inner_v1, " instead of ", inner_v2)
                    different_cells += 1
        return 1 - different_cells / total_cells
