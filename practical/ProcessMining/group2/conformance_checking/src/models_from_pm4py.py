import csv
import os
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from enum import Enum
from pm4py.objects.log.importer.xes import importer

FILE_DIR = os.path.dirname(__file__)


class AlgoPm4Py(Enum):
    ALPHA = 1
    ALPHAPLUS = 2
    INDUCTIVEMINER = 3
    HEURISTICMINER = 4


def get_log_from_file(file_path="InputLogs/L1.csv"):
    """
    Load an event log from a file.

    This function loads an event log from a specified file path. It supports both
    .csv and .xes file formats. If the file is a .xes file, it uses the `importer`
    to load the log. Otherwise, it reads a .csv file and constructs the event log manually.

    Parameters:
    ----------
    file_path : str, optional
        The path to the log file. Default is "InputLogs/L1.csv".

    Returns:
    -------
    EventLog
        The loaded event log.
    """

    file_path = os.path.join(FILE_DIR, "..", file_path)

    if file_path.endswith(".xes"):
        log = importer.apply(file_path)
    else:
        log = EventLog()

        with open(file_path, newline='') as file:
            reader = csv.reader(file)
            current_trace = None
            trace = None
            for trace_id, activity in reader:
                if current_trace != trace_id:
                    if trace is not None:
                        log.append(trace)
                    trace = Trace()
                    trace.attributes['concept:name'] = trace_id
                    current_trace = trace_id
                event = Event()
                event["concept:name"] = activity
                trace.append(event)

            if trace is not None:
                log.append(trace)
    return log


def get_model_from_pm4py(
    log,
    algorithm: AlgoPm4Py = AlgoPm4Py.ALPHAPLUS,
):
    """
    Discover a Petri net model from an event log using the specified algorithm.

    This function uses the PM4Py library to discover a Petri net model from the provided
    event log based on the selected process mining algorithm.

    Parameters:
    ----------
    log : EventLog
        The event log from which to discover the Petri net model.
    algorithm : AlgoPm4Py, optional
        The process mining algorithm to use for discovery. Default is AlgoPm4Py.ALPHAPLUS.

    Returns:
    -------
    tuple
        A tuple containing the discovered Petri net, initial marking, and final marking.
    """
    if algorithm == AlgoPm4Py.ALPHA:
        return pm4py.discover_petri_net_alpha(log)
    elif algorithm == AlgoPm4Py.ALPHAPLUS:
        return pm4py.discover_petri_net_alpha_plus(log)
    elif algorithm == AlgoPm4Py.INDUCTIVEMINER:
        return pm4py.discover_petri_net_inductive(log)
    elif algorithm == AlgoPm4Py.HEURISTICMINER:
        return pm4py.discover_petri_net_heuristics(log)
