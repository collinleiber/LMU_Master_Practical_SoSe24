from typing import Dict, Optional, Union, Tuple

from IPython.core.display import Image
from IPython.core.display_functions import display
from pm4py.objects.petri_net.obj import PetriNet, Marking
from graphviz import Digraph
import tempfile
import matplotlib.pyplot as plt
import matplotlib


class Visualizer:
    """
    A class used to visualize Process Trees and Petri Nets.

    ...

    Methods
    -------
    build_process_tree(tree, graph=None)
        Builds a process tree from a given tree structure.
    build_petri_net(net, initial_marking, final_marking, tokens=None)
        Builds a Petri net from a given net structure, initial marking, final marking, and tokens.
    display(graph)
        Displays the given graph.
    save(graph, file_name)
        Saves the given graph to a file.
    """

    def build_process_tree(self, tree: Union[Tuple, str], graph: Optional[Digraph] = None) -> Digraph:
        """
        Builds a process tree from a given tree structure.

        Parameters
        ----------
        tree : Union[Tuple, str]
            The tree structure to visualize.
        graph : Optional[Digraph], optional
            An existing graph to add to, by default None

        Returns
        -------
        Digraph
            The resulting graph.
        """
        # Create a temporary file for the graph
        filename = tempfile.NamedTemporaryFile(suffix='.gv')
        filename.close()

        # If no existing graph is provided, create a new one
        graph = graph if graph else Digraph("process tree", filename=filename.name, engine='dot')
        graph.format = 'png'

        # If the tree is a tuple, it represents an operator with children
        if isinstance(tree, tuple):
            operator, children, node_id = tree
            # Create a circular node for the operator
            graph.node(node_id, operator, shape='circle')
            for child in children:
                # If the child is a tuple, it is a subtree
                if isinstance(child, tuple):
                    child_id = child[2]
                    # Recursively add child nodes
                    self.build_process_tree(child, graph)
                else:
                    # Create a unique ID for each tau node
                    if child == '𝜏':
                        child_id = f'node{child}_{id(child)}'
                    else:
                        child_id = f'node{child}'
                    # Create a rectangular node for the activity
                    graph.node(child_id, child, shape='box', fontname="Arial")
                # Add an edge from the operator to the child node
                graph.edge(node_id, child_id, arrowhead='none')
        else:
            node_id = f'node{tree}'
            # Create a rectangular node for the activity
            graph.node(node_id, tree, shape='box')

        return graph

    def build_petri_net(self, net: PetriNet, initial_marking: Marking, final_marking: Marking,
                        tokens: Optional[Dict[str, Dict]] = None) -> Digraph:
        """
        Builds a Petri net from a given net structure, initial marking, final marking, and tokens.

        Parameters
        ----------
        net : PetriNet
            The Petri net structure to visualize.
        initial_marking : Marking
            The initial marking of the Petri net.
        final_marking : Marking
            The final marking of the Petri net.
        tokens : Optional[Dict[str, Dict]], optional
            The tokens in the Petri net, by default None

        Returns
        -------
        Digraph
            The resulting graph.
        """
        # Create a temporary file for the graph
        filename = tempfile.NamedTemporaryFile(suffix='.gv')
        filename.close()

        # Create a new graph with the name of the Petri net
        graph = Digraph(net.name, filename=filename.name, engine='dot')
        graph.graph_attr.update({'rankdir': 'LR', 'bgcolor': 'white'})
        graph.format = 'png'
        font_size = "12"

        # Add transitions to the graph
        for t in net.transitions:
            textcolor = "black"
            label = str(t.label)
            fillcolor = "transparent" if t.label else "black"
            self._create_graph_node(graph, id(t), label, style='filled', shape='box', fillcolor=fillcolor,
                                    fontsize=font_size, fontname="Arial", fontcolor=textcolor)

        # Add places to the graph
        places_sort_list = sorted(list(net.places), key=lambda x: x.name)
        for p in places_sort_list:
            fillcolor, label, penwidth, fontsize, shape = self._get_place_attributes(p, initial_marking,
                                                                                     final_marking, tokens)
            self._create_graph_node(graph, id(p), label, shape=shape, fixedsize='true', width='0.75', style="filled",
                                    fillcolor=fillcolor, fontname="Arial", weight="bold", penwidth=penwidth,
                                    fontsize=fontsize)

        # Add arcs to the graph
        arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))
        for a in arcs_sort_list:
            color, weight = self._get_edge_attributes(a.source, a.target, tokens)
            self._create_graph_edge(graph, a.source, a.target, color=color, penwidth=weight,
                                    fontsize=font_size, arrowhead="normal")

        graph.attr(overlap='false')

        return graph

    @staticmethod
    def _create_graph_node(graph: Digraph, node_id: str, label: str, **kwargs: str) -> None:
        """
        Create a node in the graph with the given label and attributes.

        Parameters
        ----------
        graph : Digraph
            The graph to add the node to.
        node_id : str
            The unique identifier of the node.
        label : str
            The label of the node.
        kwargs : str
            Additional attributes of the node.
        """
        graph.node(str(node_id), label, **kwargs)

    @staticmethod
    def _create_graph_edge(graph: Digraph, source: Union[PetriNet.Transition, PetriNet.Place],
                           target: Union[PetriNet.Transition, PetriNet.Place], **kwargs: str) -> None:
        """
        Create an edge in the graph with the given attributes.

        Parameters
        ----------
        graph : Digraph
            The graph to add the edge to.
        source
            The source node of the edge.
        target
            The target node of the edge.
        kwargs : str
            Additional attributes of the edge.
        """
        graph.edge(str(id(source)), str(id(target)), **kwargs)

    def _get_edge_attributes(self, source: Union[PetriNet.Transition, PetriNet.Place],
                             target: Union[PetriNet.Transition, PetriNet.Place],
                             tokens: Optional[Dict[str, Dict]]) -> Tuple:
        """
        Determines the color and weight of an edge in the graph based on the source and target nodes and tokens.

        Parameters
        ----------
        source : Union[PetriNet.Transition, PetriNet.Place]
            The source node of the edge.
        target : Union[PetriNet.Transition, PetriNet.Place]
            The target node of the edge.
        tokens : Optional[Dict[str, Dict]]
            The tokens in the Petri net, by default None

        Returns
        -------
        Tuple
            The color and weight of the edge.
        """
        color = "black"
        weight = "1"
        if tokens:
            missing_tokens = tokens["missing"]
            max_missing_tokens = max([v for k, v in missing_tokens.items()])
            remaining_tokens = tokens["remaining"]
            max_remaining_tokens = max([v for k, v in remaining_tokens.items()])
            all_tokens = missing_tokens | remaining_tokens

            # If either the source or target node has tokens, adjust the color and weight of the edge
            if source in all_tokens or target in all_tokens:
                missing_source = missing_tokens.get(source, 0)
                remaining_source = remaining_tokens.get(source, 0)
                missing_target = missing_tokens.get(target, 0)
                remaining_target = remaining_tokens.get(target, 0)
                missing = (missing_source + missing_target) / 2
                remaining = (remaining_source + remaining_target) / 2
                color = self._get_color(missing, remaining,
                                        max_missing_tokens, max_remaining_tokens)
                weight = "2"
        return color, weight

    def _get_place_attributes(self, place: PetriNet.Place, initial_marking: Marking,
                              final_marking: Marking, tokens: Optional[Dict[str, Dict]] = None) -> Tuple:
        """
        Determines the attributes of a place in the graph based on the place, initial marking,
        final marking, and tokens.

        Parameters
        ----------
        place : PetriNet.Place
            The place in the Petri net.
        initial_marking : Marking
            The initial marking of the Petri net.
        final_marking : Marking
            The final marking of the Petri net.
        tokens : Optional[Dict[str, Dict]]
            The tokens in the Petri net, by default None

        Returns
        -------
        Tuple
            The fill color, label, pen width, font size, and shape of the place.
        """
        fillcolor = "transparent"
        label = ""
        penwidth = "1"
        fontsize = "12"
        shape = "circle"

        # If the place is in the initial marking, adjust the label, font size, and shape
        if place in initial_marking:
            label = "<&#9679;>" if initial_marking[place] == 1 else str(initial_marking[place])
            fontsize = "34"
            shape = "circle"
        # If the place is in the final marking, adjust the label, font size, and shape
        elif place in final_marking:
            label = "<&#9632;>" if final_marking[place] == 1 else str(final_marking[place])
            fontsize = "32"
            shape = "doublecircle"

        # If there are tokens, adjust the fill color, label, pen width, and font size based on the tokens
        if tokens:
            missing_tokens = tokens["missing"]
            max_missing_tokens = max([v for k, v in missing_tokens.items()])
            remaining_tokens = tokens["remaining"]
            max_remaining_tokens = max([v for k, v in remaining_tokens.items()])

            if place in remaining_tokens and place in missing_tokens:
                fillcolor = self._get_color(missing_tokens[place], remaining_tokens[place],
                                            max_missing_tokens, max_remaining_tokens)
                label = f'<<B>+{remaining_tokens[place]}<BR></BR>-{missing_tokens[place]}</B>>'
                if place in initial_marking:
                    label = f'<<FONT POINT-SIZE="28">&#9679;</FONT><BR></BR><B>+{remaining_tokens[place]} -{missing_tokens[place]}</B>>'
                penwidth = "2"
                fontsize = "12"
            elif place in remaining_tokens:
                fillcolor = self._get_color(0, remaining_tokens[place],
                                            max_missing_tokens, max_remaining_tokens)
                label = f'<<B>+{remaining_tokens[place]}</B>>'
                if place in initial_marking:
                    label = f'<<FONT POINT-SIZE="28">&#9679;</FONT><BR></BR><B>+{remaining_tokens[place]}</B>>'
                penwidth = "2"
                fontsize = "12"
            elif place in missing_tokens:
                fillcolor = self._get_color(missing_tokens[place], 0,
                                            max_missing_tokens, max_remaining_tokens)
                label = f'<<B>-{missing_tokens[place]}</B>>'
                if place in initial_marking:
                    label = f'<<FONT POINT-SIZE="28">&#9679;</FONT><BR></BR><B>-{missing_tokens[place]}</B>>'
                penwidth = "2"
                fontsize = "12"

        return fillcolor, label, penwidth, fontsize, shape

    @staticmethod
    def _get_color(missing_tokens: int, remaining_tokens: int, max_tokens_miss: int,
                   max_tokens_remain: int, scale: float = 0.6) -> str:
        """
        Determines the color of a node in the graph based on the number of missing and remaining tokens.

        Parameters
        ----------
        missing_tokens : int
            The number of missing tokens.
        remaining_tokens : int
            The number of remaining tokens.
        max_tokens_miss : int
            The maximum number of missing tokens.
        max_tokens_remain : int
            The maximum number of remaining tokens.
        scale : float, optional
            The scale factor for the color map, by default 0.6

        Returns
        -------
        str
            The color of the node.
        """
        res = remaining_tokens - missing_tokens
        # Default color (i.e., balance between missing and remaining tokens is zero)
        fillcolor = "Thistle"
        if res > 0:
            # If there are more remaining tokens than missing tokens, use a blue color map
            ratio = res / max_tokens_remain
            ratio = ratio * scale + (1 - scale) / 2
            cmap = plt.cm.get_cmap("Blues")
            fillcolor = cmap(ratio)
            fillcolor = matplotlib.colors.to_hex(fillcolor)
        elif res < 0:
            # If there are more missing tokens than remaining tokens, use a red color map
            ratio = res / max_tokens_miss
            ratio = abs(ratio) * scale + (1 - scale) / 2
            cmap = plt.cm.get_cmap("Reds")
            fillcolor = cmap(ratio)
            fillcolor = matplotlib.colors.to_hex(fillcolor)
        return fillcolor

    @staticmethod
    def display(graph: Digraph) -> None:
        """
        Displays the given graph in a Jupyter Notebook.

        Parameters
        ----------
        graph : Digraph
            The graph to display.
        """
        display(Image(graph.render()))

    @staticmethod
    def save(graph: Digraph, file_name: str) -> None:
        """
        Saves the given graph to a PNG file.

        Parameters
        ----------
        graph : Digraph
            The graph to save.
        file_name : str
            The name of the file to save the graph to.
        """
        graph.render(filename=file_name, format='png', cleanup=True)
