import graphviz
from alphaminer import AlphaMiner


def visualize_full_alpha_miner(input_event_log_path: str, output_path: str):
    """
    Visualize the process graph using Graphviz based on all relations identified by Alpha Miner.

    Parameters:
        input_event_log_path (str): Path to the event log file.
        output_path (str): Path to save the output graph image.
    """
    # Analyze the log file using Alpha Miner
    alpha_miner = AlphaMiner(input_event_log_path)

    # Initialize a Graphviz directed graph object
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR')  # Set left-to-right direction

    # Add all activities as nodes
    for activity_id, activity_name in alpha_miner.activities.items():
        dot.node(activity_name)

    # Add sequential pairs as edges
    for a1, a2 in alpha_miner.sequential_pairs:
        dot.edge(alpha_miner.activities[a1], alpha_miner.activities[a2], label='Sequential', color='blue')

    # Add parallel pairs as bidirectional edges
    for a1, a2 in alpha_miner.parallel_pairs:
        dot.edge(alpha_miner.activities[a1], alpha_miner.activities[a2], label='Parallel', color='green')
        dot.edge(alpha_miner.activities[a2], alpha_miner.activities[a1], label='Parallel', color='green')

    # Add before pairs as edges
    for a1, a2 in alpha_miner.before_pairs:
        dot.edge(alpha_miner.activities[a2], alpha_miner.activities[a1], label='Before', color='red')

    # Add not-following pairs as dotted edges
    for a1, a2 in alpha_miner.not_following_pairs:
        dot.edge(alpha_miner.activities[a1], alpha_miner.activities[a2], label='Not Followed', style='dotted',
                 color='gray')

    # Render and save the graph
    dot.render(output_path, view=True)


if __name__ == "__main__":
    # path to the log file
    event_log_path = "example_files/running-example.csv"
    output_graph_path = "full_alpha_miner_graph"
    visualize_full_alpha_miner(event_log_path, output_graph_path)
