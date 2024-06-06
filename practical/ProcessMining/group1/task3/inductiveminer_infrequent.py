from typing import Optional, List, Tuple, Dict, Set

from inductiveminer import InductiveMiner, CutType


class InductiveMinerInfrequent(InductiveMiner):
    def __init__(self, event_log: Optional[List[Tuple[str]]] = None, threshold: float = 0.0):
        super().__init__(event_log=event_log)
        self.threshold = threshold

    def run(self) -> None:
        # Initialize the list of sublogs with the original event log
        sublogs = [self.event_log]

        # Iterate over the sublogs until the list is empty
        while len(sublogs) > 0:
            log = sublogs[0]

            # Update the directly-follows graph (dfg), start_activities, and end_activities for the current sublog
            dfg, start_activities, end_activities = self._get_dfg(log)

            # Check for base cases and build the corresponding part of the process tree
            base_cut, operator = self._handle_base_cases(log)
            if base_cut:  # If not a base case, apply different types of cuts
                self._build_process_tree(base_cut, operator)
            else:  # try to split log based on operator and build corresponding part of the process tree
                groups, operator = self._apply_cut(log, dfg, start_activities, end_activities)
                # Add the new sublogs to the list if not fall through case
                if operator != CutType.NONE:
                    new_sublogs = self._split_log(log, groups)
                    sublogs.extend(new_sublogs)
                    # Build the corresponding part of the process tree
                    self._build_process_tree(groups, operator)
                else:  # If fall through case, apply infrequent logic TODO refactor for super call
                    base_cut, operator = self._handle_base_cases_filtered(log)
                    if base_cut:  # If not a base case, apply different types of cuts
                        self._build_process_tree(base_cut, operator)
                    else:  # try to split log based on operator and build corresponding part of the process tree
                        groups, operator = self._apply_cut_filtered(log, dfg, start_activities, end_activities)
                        if operator != CutType.NONE:  # If not fall through case
                            new_sublogs = self._split_log_filtered(log, groups, operator)  # Apply IMi filters
                            sublogs.extend(new_sublogs)
                            self._build_process_tree(groups, operator)
                        else:  # If fall through case, apply flower model
                            self._build_process_tree(groups, CutType.LOOP)

            # Remove the old sublog from the list
            sublogs.remove(log)

        dfg, dfg_start, dfg_end = self.get_frequent_directly_follows_graph()
        edfg, edfg_start, edfg_end = self.get_frequent_eventually_follows_graph()
    def _apply_cut_filtered(self, log: List[Tuple[str]], dfg: Dict[Tuple[str, str], int],
                            start_activities: Dict[str, int], end_activities: Dict[str, int]) \
            -> Tuple[List[Set[str]], CutType]:


        # TODO refactor apply_cut to make super call possible instead of duplicate code
        # super()._apply_cut(log=log, dfg=dfg, start_activities=dfg_start, end_activities=dfg_end)

        # Try to apply different types of cuts to the current sublog
        sequence_cut = self._sequence_cut(efg_filtered, start_activities, end_activities)
        xor_cut = self._xor_cut(dfg_filtered, start_activities, end_activities)
        parallel_cut = self._parallel_cut(dfg_filtered, start_activities, end_activities)
        loop_cut = self._loop_cut(dfg_filtered, start_activities, end_activities)

        # If a nontrivial cut (>1) is found, return the partition and the corresponding operator
        if self._is_nontrivial(sequence_cut):
            return sequence_cut, CutType.SEQUENCE
        elif self._is_nontrivial(xor_cut):
            return xor_cut, CutType.XOR
        elif self._is_nontrivial(parallel_cut):
            return parallel_cut, CutType.PARALLEL
        elif self._is_nontrivial(loop_cut):
            return loop_cut, CutType.LOOP
        else:  # If no nontrivial cut is found, apply the fall-through case (flower model)
            flower_groups = self._handle_fall_through(log)
            return flower_groups, CutType.NONE

    def _handle_base_cases_filtered(self, log: List[Tuple[str]]) -> Tuple[List[Set[str]], CutType]:
        # TODO: Apply IMi filters on base cases
        pass

    def get_frequent_directly_follows_graph(self) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
        pass

    def _calculate_eventual_follows_graph(self):
        pass

    def get_frequent_eventually_follows_graph(self) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
        pass

    def _split_log_filtered(self, log: List[Tuple[str]], groups: List[Set[str]],
                            operator: CutType) -> List[List[Tuple[str]]]:
        # TODO: Apply IMi filters on log splitting based on operator
        return []

    def xor_split_infrequent(self):
        pass

    def sequence_split_infrequent(self):
        pass

    def loop_split_infrequent(self):
        pass

