import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.discovery.inductive import algorithm as inductive_miner


def log_to_process_tree(log_dict):
    event_log = EventLog()

    for case_id, count in log_dict.items():
        for _ in range(count):
            trace = Trace()
            for event in case_id:
                trace.append(Event({"concept:name": event}))
            event_log.append(trace)

    return event_log


def get_process_tree_with_pm4py(log):
    tree = inductive_miner.apply(log_to_process_tree(log))
    return tree


logs = [
    ("=========== Example 1 ===========", {'abcd': 1, 'ad': 1}),
    ("=========== Example 2 ===========", {'abcd': 1, 'ad': 1, 'aefd': 1}),
    ("=========== Example 3 ===========", {'abcd': 1, 'abcd': 1, 'aed': 1}),
    ("=========== L1 ===========", {'abcd': 3, 'acbd': 2, 'aed': 1}),
    (
        "=========== L2 ===========",
        {
            'abcd': 3,
            'acbd': 4,
            'abcefbcd': 2,
            'abcefcbd': 1,
            'acbefbcd': 2,
            'acbefbcefcbd': 1,
        },
    ),
    (
        "=========== L3 ===========",
        {'abcdefbdceg': 1, 'abdceg': 2, 'abcdefbcdefbdceg': 1},
    ),
    ("=========== L4 ===========", {'acd': 45, 'bcd': 42, 'ace': 38, 'bce': 22}),
    (
        "=========== L5 ===========",
        {'abef': 2, 'abecdbf': 3, 'abcedbf': 2, 'abcdebf': 4, 'aebcdbf': 3},
    ),
    ("=========== L6 ===========", {'aceg': 2, 'aecg': 3, 'bdfg': 2, 'bfdg': 4}),
    ("=========== L7 ===========", {'ac': 2, 'abc': 3, 'abbc': 2, 'abbbbc': 1}),
    ("=========== L8 ===========", {'abd': 3, 'abcbd': 2, 'abcbcbd': 1}),
    ("=========== L9 ===========", {'acd': 45, 'bce': 42}),
    ("=========== L10 ===========", {'aa': 55}),
    ("=========== L11 ===========", {'abc': 20, 'ac': 30}),
    ("=========== L12 ===========", {'acd': 45, 'bce': 42, 'ace': 20}),
    (
        "=========== L13 ===========",
        {'abcde': 10, 'adbe': 10, 'aeb': 1, 'acb': 1, 'abdec': 2},
    ),
    ("=========== L14 ===========", {'abcd': 10, 'dab': 10, 'adc': 10, 'bcd': 5}),
    (
        "=========== L15 ===========",
        {'ab': 25, 'ac': 25, 'db': 25, 'dc': 25, 'abac': 1},
    ),
    ("=========== L16 ===========", {'abcd': 20, 'ad': 20}),
    (
        "=========== L17 ===========",
        {'abcde': 10, 'adbe': 10, 'aeb': 10, 'acb': 10, 'abdec': 10, 'cad': 1},
    ),
    (
        "=========== L18 ===========",
        {
            'abcgeh': 10,
            'abcfgh': 10,
            'abdgeh': 10,
            'abdeg': 10,
            'acbegh': 10,
            'acbfgh': 10,
            'adbegh': 10,
            'adbfgh': 10,
        },
    ),
    ("=========== L19 ===========", {'abcdfe': 1, 'acbdfe': 1, 'acbde': 1, 'abcde': 1}),
]

output = ""
for title, event_log in logs:
    output += "{}\n{}\n".format(title, str(get_process_tree_with_pm4py(event_log)))

with open('./output/pm4py_process_trees.txt', 'w') as file:
    file.write(output)
