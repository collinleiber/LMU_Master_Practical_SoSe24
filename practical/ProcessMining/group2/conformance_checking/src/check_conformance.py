import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from src.generate_footprint import (
    FootPrintMatrix,
)


class ConformanceChecking:
    # Initialize ConformanceChecking Object
    def __init__(self, fpm_1, fpm_2):
        self.fpm_1 = fpm_1
        self.fpm_2 = fpm_2

    # Checks two dictionaries (footprint)
    # TODO only works if keys of dicts are the same
    def get_conformance_matrix(self):
        dict_out = {}
        for (outer_k1, outer_v1), (outer_k2, outer_v2) in zip(
            self.fpm_1.relations.items(), self.fpm_2.relations.items()
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

    def get_conformance_value(self):
        different_cells = 0
        total_cells = len(self.fpm_1.relations) ** 2
        for (outer_k1, outer_v1), (outer_k2, outer_v2) in zip(
            self.fpm_1.relations.items(), self.fpm_2.relations.items()
        ):
            inner_dict_out = {}
            for (inner_k1, inner_v1), (inner_k2, inner_v2) in zip(
                outer_v1.items(), outer_v2.items()
            ):
                if inner_v1 != inner_v2:
                    different_cells += 1

        return 1 - different_cells / total_cells
    

    def pipeline(self, log, algorithm): # TODO: type hinting
        # Pipeline:
        # 1. Get footprint matrix from log
        # 2. Run algorithm on log
        # 3. Replay resulting model from algorithm
        # 4. Get footprint matrix from replayed log
        pass

    def log_2_log(self, log, algorithm): # TODO: type hinting
        """
        This function takes a log and an algorithm, splits it into multiple sublogs. 
        Next, after running the algorithm on each sublog, the resulting model is being replayed in order to get a new log.
        Finally, the resulting logs are being compared to the original log.
        """
        # Convert the log to csv for easier splitting
        # log_csv = pm4py.something(log)

        # Split log into sublogs
        sublogs = []
        # Use https://pm4py-source.readthedocs.io/en/stable/pm4py.objects.log.util.html#pm4py.objects.log.util.split_train_test.split
        # Set k = 4 for now
        # train, test = pm4py.objects.log.util.split_train_test.split(log_csv, train_percentage=0.5)
        # train, test = pm4py.objects.log.util.split_train_test.split(train, train_percentage=0.5)
        # sublogs.append(train)
        # sublogs.append(test)
        # train, test = pm4py.objects.log.util.split_train_test.split(test, train_percentage=0.5)
        # sublogs.append(train)
        # sublogs.append(test)


        replayed_logs = []
        # Run algorithm on each sublog
        for sublog in sublogs:
            # Run pipeline
            # result = self.pipeline(sublog, algorithm)
            # results.append(result)
            pass

        for log in replayed_logs:
            # Compare with original log
            pass

    def log_2_model(self, log, algorithm): # TODO: type hinting
        """
        This function takes a log and an algorithm.
        The algorithm is being run on the log in order to get a model, which is then being replayed in order to get a new log.
        Finally, the resulting log is being compared to the original log.
        """
        # Run pipeline
        # result = self.pipeline(log, algorithm)
        # Compare with original log
        pass

    def model_2_model(self, log, algorithms, scenario): # TODO: type hinting
        """
        This function takes a log, a list of algorithms and one of two scenarios.
        The algorithms are being run on the log in order to get multiple models, which are then being replayed in order to get new logs.
        The resulting logs are compared in dfferent ways, depending on the scenario.
        Scenario 1: Each log is compared to the original log.
        Scenario 2: Each log is compared to the other logs.
        """
        # Run pipeline for each algorithm
        # Check for scenario
        # Compare logs accordingly
        pass
