import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from src.generate_footprint import (
    FootPrintMatrix,
)


class ConformanceChecking:

    # Checks two dictionaries (footprint)
    def get_conformance_matrix(self, fpm_1, fpm_2):
        dict_out = {}
        keys_1 = set(fpm_1.relations.keys())
        keys_2 = set(fpm_2.relations.keys())
        all_keys = keys_1.union(keys_2)

        for key in all_keys:
            inner_dict_out = {}
            inner_keys_1 = set(fpm_1.relations.get(key, {}).keys())
            inner_keys_2 = set(fpm_2.relations.get(key, {}).keys())
            all_inner_keys = inner_keys_1.union(inner_keys_2)

            for inner_key in all_inner_keys:
                inner_v1 = fpm_1.relations.get(key, {}).get(inner_key, '')
                inner_v2 = fpm_2.relations.get(key, {}).get(inner_key, '')
                if inner_v1 == inner_v2:
                    inner_dict_out[inner_key] = ''
                else:
                    inner_dict_out[inner_key] = '{}:{}'.format(inner_v1, inner_v2)

            dict_out[key] = inner_dict_out

        return FootPrintMatrix.from_relations(dict_out)

    def get_conformance_value(self, fpm_1, fpm_2):
        different_cells = 0
        keys_1 = set(fpm_1.relations.keys())
        keys_2 = set(fpm_2.relations.keys())
        all_keys = keys_1.union(keys_2)
        total_cells = len(all_keys) ** 2

        for key in all_keys:
            inner_keys_1 = set(fpm_1.relations.get(key, {}).keys())
            inner_keys_2 = set(fpm_2.relations.get(key, {}).keys())
            all_inner_keys = inner_keys_1.union(inner_keys_2)

            for inner_key in all_inner_keys:
                inner_v1 = fpm_1.relations.get(key, {}).get(inner_key, '')
                inner_v2 = fpm_2.relations.get(key, {}).get(inner_key, '')
                if inner_v1 != inner_v2:
                    different_cells += 1

        return 1 - different_cells / total_cells if total_cells > 0 else 0.0
