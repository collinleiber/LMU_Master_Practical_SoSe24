import csv
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event


def get_petri_net_from_pm4py_alpha_miner(file_path='./../Logs/L2.csv'):

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

    return pm4py.discover_petri_net_alpha_plus(log)
