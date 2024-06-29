from models_from_pm4py import (
    get_log_from_file,
    get_model_from_pm4py,
    AlgoPm4Py,
)
from replay import get_traces_with_replay
from generate_footprint import FootPrintMatrix
from check_conformance import ConformanceChecking
from visualize_matrix import visualize_sorted_dict
from pm4py.visualization.petri_net import visualizer
from itertools import combinations


class Comparison:

    def pipeline(self, input_log, algorithm: AlgoPm4Py):
        """
        input_log is on form of dictionary like the one FootPrintMatrix uses.
        """
        # Pipeline:
        # 1. Get footprint matrix from log
        fpm_original = FootPrintMatrix(input_log)
        fpm_original.generate_footprint()
        # 2. Run algorithm on log
        net, start, end = get_model_from_pm4py(input_log, algorithm)
        # 3. Replay resulting model from algorithm
        logs_replayed = get_traces_with_replay(net, start, end, 150)
        # 4. Get footprint matrix from replayed log
        fpm_replayed = FootPrintMatrix(logs_replayed)
        fpm_replayed.generate_footprint()
        return fpm_original, fpm_replayed

    def log_2_log(self, log, algorithm):  # TODO: type hinting
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

    def log_2_model(self, log, algorithm):  # TODO: type hinting
        """
        This function takes a log and an algorithm.
        The algorithm is being run on the log in order to get a model, which is then being replayed in order to get a new log.
        Finally, the resulting log is being compared to the original log.
        """
        # Run pipeline
        # result = self.pipeline(log, algorithm)
        # Compare with original log
        pass

    def model_2_model(self, log, algorithms, scenario):  # TODO: type hinting
        """
        This function takes a log, a list of algorithms and one of two scenarios.
        The algorithms are being run on the log in order to get multiple models, which are then being replayed in order to get new logs.
        The resulting logs are compared in dfferent ways, depending on the scenario.
        Scenario 1: Each log is compared to the original log.
        Scenario 2: Each log is compared to the other logs.
        """

        log_from_file = get_log_from_file(log)

        generated_fpms = {}
        fpm_original = None
        for algorithm in algorithms:
            pipeline_fpm_original, pipeline_fpm_generated = self.pipeline(
                log_from_file, algorithm
            )
            # visualize_sorted_dict(pipeline_fpm_generated.relations, "m2m_{}".format(str(algorithm)))

            generated_fpms[str(algorithm)] = pipeline_fpm_generated
            if fpm_original == None:
                fpm_original = pipeline_fpm_original
                # visualize_sorted_dict(pipeline_fpm_original.relations, "m2m_original")

        comparison_values = {}
        if scenario == 1:
            for algorithm, fpm_from_algorithm in generated_fpms.items():
                key = "InputLog vs. {}".format(str(algorithm))
                print("Checking Conformance Value of: {}".format(key))
                conformance_checking = ConformanceChecking()
                comparison_values[key] = conformance_checking.get_conformance_value(
                    fpm_original, fpm_from_algorithm
                )

        if scenario == 2:
            for (algo1, fpm1), (algo2, fpm2) in combinations(generated_fpms.items(), 2):
                key = "{} vs. {}".format(algo1, algo2)
                print("Checking Conformance Value of: {}".format(key))
                conformance_checking = ConformanceChecking()
                comparison_values[key] = conformance_checking.get_conformance_value(
                    fpm1, fpm2
                )

        return comparison_values


log_path = "InputLogs/L5.csv"


com = Comparison()
print(com.model_2_model(log_path, [AlgoPm4Py.ALPHA, AlgoPm4Py.ALPHAPLUS], 1))
