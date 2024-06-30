import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from typing import Dict, List, Tuple

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
from pm4py.objects.log.util import split_train_test
from itertools import combinations
from pm4py.algo.simulation.playout.petri_net import variants


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
        #print("net: ",algorithm,net)
        gviz = visualizer.apply(net,start,end)
        gviz.graph_attr['label'] = str(algorithm)
        gviz.graph_attr['labelloc'] = 't' 
        visualizer.view(gviz)
        # 3. Replay resulting model from algorithm
        logs_replayed = get_traces_with_replay(net, start, end, variants.extensive)
        # 4. Get footprint matrix from replayed log
        fpm_replayed = FootPrintMatrix(logs_replayed)
        fpm_replayed.generate_footprint()
        return fpm_original, fpm_replayed

    def log_2_log(self, event_log: str, algorithm: 'AlgoPm4Py') -> List[Tuple[str, float]]:
        """
        This function takes a log and an algorithm, splits it into multiple sublogs.
        Next, after running the algorithm on each sublog, the resulting model is being replayed in order to get a new log.
        Finally, the resulting logs are being compared to the original log.

        :param event_log: Path to the event log.
        :param algorithm: Algorithm to be used.
        :return: List of tuples, where each tuple contains the name of the sublog and the conformance value.
        """
        def split_log_into_sublogs(log):
            sublogs = []
            # Initial split to divide the log into two equal parts
            split_1, split_2 = split_train_test.split(log, train_percentage=0.5)
            
            # Further split each part into two sublogs
            sublog_1, sublog_2 = split_train_test.split(split_1, train_percentage=0.5)
            sublog_3, sublog_4 = split_train_test.split(split_2, train_percentage=0.5)
            
            sublogs.extend([sublog_1, sublog_2, sublog_3, sublog_4])
            return sublogs


        log = get_log_from_file(event_log)

        # Split log into sublogs. We use 4 sublogs for ease of use.
        sublogs = split_log_into_sublogs(log)

        replayed_logs = []
        # Run algorithm on each sublog
        for sublog in sublogs:
            # Run pipeline
            footprint_of_log, footprint_of_replayed_log = self.pipeline(sublog, algorithm)
            replayed_logs.append(footprint_of_replayed_log)

        # Compare with original log
        comparison_values = []
        conformance_checking = ConformanceChecking()
        for i, footprint_of_replayed_log in enumerate(replayed_logs):
            result = conformance_checking.get_conformance_value(footprint_of_log, footprint_of_replayed_log)
            comparison_values.append((f"sublog_{i}", result))

        return comparison_values

    def log_2_model(self, event_log: str, algorithm: 'AlgoPm4Py') -> float:
        """
        This function takes a log and an algorithm.
        The algorithm is being run on the log in order to get a model, which is then being replayed in order to get a new log.
        Finally, the resulting log is being compared to the original log.

        :param event_log: Path to the event log.
        :param algorithm: Algorithm to be used.
        :return: Conformance value.
        """
        log = get_log_from_file(event_log)

        # Run pipeline
        footprint_of_log, footprint_of_replayed_log = self.pipeline(log, algorithm)
        # Compare with original log
        conformance_checking = ConformanceChecking()
        result = conformance_checking.get_conformance_value(footprint_of_log, footprint_of_replayed_log)

        return result

    def model_2_model(self, log: str, algorithms: List['AlgoPm4Py'], scenario: int) -> Dict[str, float]:
        """
        This function takes a log, a list of algorithms and one of two scenarios.
        The algorithms are being run on the log in order to get multiple models, which are then being replayed in order to get new logs.
        The resulting logs are compared in dfferent ways, depending on the scenario.
        Scenario 1: Each log is compared to the original log.
        Scenario 2: Each log is compared to the other logs.

        :param log: Path to the event log.
        :param algorithms: List of algorithms to be used.
        :param scenario: Scenario to be used.
        :return: Dictionary containing the comparison values.
        """

        log_from_file = get_log_from_file(log)

        generated_fpms = {}
        fpm_original = None
        for algorithm in algorithms:
            pipeline_fpm_original, pipeline_fpm_generated = self.pipeline(
                log_from_file, algorithm
            )
            visualize_sorted_dict(pipeline_fpm_generated.relations, "m2m_{}".format(str(algorithm)))

            generated_fpms[str(algorithm)] = pipeline_fpm_generated
            if fpm_original == None:
                fpm_original = pipeline_fpm_original
                visualize_sorted_dict(pipeline_fpm_original.relations, "m2m_original")

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


log_path = "InputLogs/L4.csv"


com = Comparison()
print(com.model_2_model(log_path, [AlgoPm4Py.ALPHA, AlgoPm4Py.ALPHAPLUS, AlgoPm4Py.HEURISTICMINER, AlgoPm4Py.INDUCTIVEMINER], 1))
