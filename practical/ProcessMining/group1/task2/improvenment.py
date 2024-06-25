from collections import defaultdict
import pandas as pd
import os
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

def create_directly_follows_graph(logs):
    directly_follows = defaultdict(set)

    for case in logs:
        for i in range(len(case) - 1):
            directly_follows[case[i]].add(case[i+1])
            # 2-length loops
            if i < len(case) - 2 and case[i] == case[i+2]:
                directly_follows[case[i]].add(case[i+1])
                directly_follows[case[i+1]].add(case[i])
        # self loop
        for i in range(len(case)):
            if i < len(case) - 1 and case[i] == case[i + 1]:
                directly_follows[case[i]].add(case[i])

    return directly_follows

def identify_short_loops(directly_follows):
    loops = {
        'self_loops': [],
        'length_2_loops': []
    }

    for activity, followers in directly_follows.items():
        if activity in followers:
            loops['self_loops'].append(activity)

    for activity, followers in directly_follows.items():
        for follower in followers:
            if activity in directly_follows.get(follower, []):
                if (follower, activity) not in loops['length_2_loops'] and (activity, follower) not in loops['length_2_loops']:
                    loops['length_2_loops'].append((activity, follower))

    return loops


# remove loops
def remove_cycles_from_logs(logs, loops):
    modified_logs = []

    for activities in logs:
        new_activities = []
        i = 0
        while i < len(activities):
            # check self loops
            if i < len(activities) - 1 and activities[i] in loops['self_loops'] and activities[i] == activities[i + 1]:
                while i < len(activities) - 1 and activities[i] == activities[i + 1]:
                    i += 1

            # check 2-length loops
            elif i < len(activities) - 2 and (activities[i], activities[i + 1]) in loops['length_2_loops'] and activities[i] == activities[i + 2]:
                while i < len(activities) - 2 and activities[i] == activities[i + 2]:
                    i += 2

            new_activities.append(activities[i])
            i += 1

        modified_logs.append(new_activities)

    return modified_logs


def generate_petri_net(directly_follows, loops):
    net = PetriNet("Generated Petri Net")
    places = {}
    transitions = {}

    for activity in directly_follows:
        if activity not in transitions:
            transitions[activity] = PetriNet.Transition(activity, activity)
            net.transitions.add(transitions[activity])

        for follower in directly_follows[activity]:
            if follower not in transitions:
                transitions[follower] = PetriNet.Transition(follower, follower)
                net.transitions.add(transitions[follower])

            place_id = (activity, follower)
            if place_id not in places:
                places[place_id] = PetriNet.Place(str(place_id))
                net.places.add(places[place_id])

            add_arc_from_to(transitions[activity], places[place_id], net)
            add_arc_from_to(places[place_id], transitions[follower], net)

    #add loops need to be tested!!!!!!
    # todo
    for activity in loops['self_loops']:
        place = PetriNet.Place("self_loop_" + activity)
        net.places.add(place)
        add_arc_from_to(transitions[activity], place, net)
        add_arc_from_to(place, transitions[activity], net)

    for (activity1, activity2) in loops['length_2_loops']:
        place1 = PetriNet.Place(f"loop_{activity1}_{activity2}_1")
        place2 = PetriNet.Place(f"loop_{activity1}_{activity2}_2")
        net.places.add(place1)
        net.places.add(place2)
        add_arc_from_to(transitions[activity1], place1, net)
        add_arc_from_to(place1, transitions[activity2], net)
        add_arc_from_to(transitions[activity2], place2, net)
        add_arc_from_to(place2, transitions[activity1], net)

    return net

# logs = [
#      ['a', 'c'],
#      ['a', 'b', 'c'],
#      ['a', 'b', 'b', 'c'],
#      ['a', 'b', 'b', 'b', 'c']
# ]

# logs = [
#      ['a', 'b', 'd'],
#      ['a', 'b', 'c', 'b', 'd'],
#      ['a', 'b', 'c', 'b', 'c', 'b', 'd']
# ]

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'example_files', 'limitation-example3.csv')
data = pd.read_csv(file_path, sep=';')
data['timestamp'] = pd.to_datetime(data['timestamp'])

grouped = data.groupby('case_id')['activity'].apply(list)
logs = grouped.tolist()
print('logs ======', logs)

dfg = create_directly_follows_graph(logs)
print('dfg ======', dict(dfg))
identified_loops = identify_short_loops(dfg)
print("Identified Loops ================", identified_loops)

modified_logs = remove_cycles_from_logs(logs, identified_loops)
print("Modified Logs = ", modified_logs)

petri_net = generate_petri_net(dfg,identified_loops)
print("petri net - ", petri_net)