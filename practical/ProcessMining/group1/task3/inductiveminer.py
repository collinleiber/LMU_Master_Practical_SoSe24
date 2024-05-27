from typing import Tuple, Dict, Set
import pm4py
from practical.ProcessMining.group1.shared.utils import read_txt_test_logs, event_log_to_dataframe


class InductiveMiner:
    def __init__(self):
        pass

    def _parallel_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int], end: Dict[str, int]) -> Set[str]:
        parallel_cut = set()
        for edge in dfg.keys():
            if all(_ in start for _ in edge) and all(_ in end for _ in edge):
                if edge[::-1] in dfg:
                    parallel_cut.update(edge)
        return parallel_cut


if __name__ == '__main__':
    log = [('a', 'b'), ('a', 'b'), ('b', 'a')]
    dfg, start, end = pm4py.discover_dfg(pm4py.format_dataframe(event_log_to_dataframe(log), case_id='case_id',
                                                                activity_key='activity', timestamp_key='timestamp'))

    print(log, dfg, start, end)

    miner = InductiveMiner()
    print(miner._parallel_cut(dfg, start, end))