import os
from typing import Dict

script_dir = os.path.dirname(os.path.abspath(__file__))

class EventLog:
    def __init__(self, traces: Dict[str, int]):
        """
        Initialize EventLog object.

        :param traces: Dictionary where keys are traces and values are counts.
        """
        self.traces = traces

    @classmethod
    def from_file(cls, file_path: str = None) -> 'EventLog':
        """
        Create an EventLog object from a file.

        :param file_path: Path to the file containing traces.
        :return: EventLog object.
        """
        with open(os.path.join(script_dir, file_path), 'r') as file:
            traces = {}
            for line in file:
                trace = line.strip()
                if trace in traces:
                    traces[trace] += 1
                else:
                    traces[trace] = 1
            return EventLog(traces)
