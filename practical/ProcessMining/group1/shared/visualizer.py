from typing import Dict, Optional

from IPython.core.display import Image
from IPython.core.display_functions import display
from pm4py.objects.petri_net.obj import PetriNet, Marking
from graphviz import Digraph
import tempfile


class Visualizer:

    def get_process_tree(self, tree, graph=None) -> Digraph:
        filename = tempfile.NamedTemporaryFile(suffix='.gv')
        filename.close()

        graph = graph if graph else Digraph("process tree", filename=filename.name, engine='dot')
        graph.format = 'png'

        if isinstance(tree, tuple):
            operator, children, node_id = tree
            graph.node(node_id, operator, shape='circle')  # Create a circular node for the operator
            for child in children:
                if isinstance(child, tuple):
                    child_id = child[2]
                    self.get_process_tree(child, graph)  # Recursively add child nodes
                else:
                    # Create a unique ID for each τ
                    if child == '𝜏':
                        child_id = f'node{child}_{id(child)}'
                    else:
                        child_id = f'node{child}'
                    graph.node(child_id, child, shape='box',
                               fontname="Arial")  # Create a rectangular node for the activity
                graph.edge(node_id, child_id, arrowhead='none')  # Add an edge from the operator to the child node
        else:
            node_id = f'node{tree}'
            graph.node(node_id, tree, shape='box')  # Create a rectangular node for the activity
        return graph

    def get_petri_net(self, net: PetriNet, initial_marking: Marking, final_marking: Marking,
                      tokens: Optional[Dict[str, Dict]] = None) -> Digraph:
        filename = tempfile.NamedTemporaryFile(suffix='.gv')
        filename.close()

        graph = Digraph(net.name, filename=filename.name, engine='dot')
        graph.graph_attr.update({'rankdir': 'LR', 'bgcolor': 'white'})
        graph.format = 'png'
        font_size = "12"

        # transitions
        for t in net.transitions:
            textcolor = "black"
            label = str(t.label)
            fillcolor = "transparent" if t.label else "black"
            self._create_graph_node(graph, id(t), label, style='filled', shape='box', fillcolor=fillcolor,
                                    fontsize=font_size, fontname="Arial", fontcolor=textcolor)

        # places
        places_sort_list = sorted(list(net.places), key=lambda x: x.name)
        for p in places_sort_list:
            fillcolor, label, penwidth, fontsize, shape = self._get_place_attributes(p, initial_marking,
                                                                                     final_marking, tokens)
            self._create_graph_node(graph, id(p), label, shape=shape, fixedsize='true', width='0.75', style="filled",
                                    fillcolor=fillcolor, fontname="Arial", weight="bold", penwidth=penwidth,
                                    fontsize=fontsize)

        # arcs
        arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))
        for a in arcs_sort_list:
            self._create_graph_edge(graph, id(a.source), id(a.target), fontsize=font_size, arrowhead="normal")

        graph.attr(overlap='false')

        return graph

    @staticmethod
    def _create_graph_node(graph, node_id, label, **kwargs):
        graph.node(str(node_id), label, **kwargs)

    @staticmethod
    def _create_graph_edge(graph, source_id, target_id, **kwargs):
        graph.edge(str(source_id), str(target_id), **kwargs)

    @staticmethod
    def _get_place_attributes(place, initial_marking: Marking, final_marking: Marking,
                              tokens: Optional[Dict[str, Dict]] = None) -> tuple:
        fillcolor = "transparent"
        label = ""
        penwidth = "1"
        fontsize = "12"
        shape = "circle"

        if place in initial_marking:
            label = "<&#9679;>" if initial_marking[place] == 1 else str(initial_marking[place])
            fontsize = "34"
            shape = "circle"
        elif place in final_marking:
            label = "<&#9632;>" if final_marking[place] == 1 else str(final_marking[place])
            fontsize = "32"
            shape = "doublecircle"

        if tokens:
            missing_tokens = tokens["missing"]
            remaining_tokens = tokens["remaining"]
            if place in remaining_tokens and place in missing_tokens:
                fillcolor = "plum"
                label = f'<<B>+{remaining_tokens[place]}<BR></BR>-{missing_tokens[place]}</B>>'
                if place in initial_marking:
                    label = f'<<FONT POINT-SIZE="28">&#9679;</FONT><BR></BR><B>+{remaining_tokens[place]} -{missing_tokens[place]}</B>>'
                penwidth = "2"
                fontsize = "12"
            elif place in remaining_tokens:
                fillcolor = "lightskyblue"
                label = f'<<B>+{remaining_tokens[place]}</B>>'
                if place in initial_marking:
                    label = f'<<FONT POINT-SIZE="28">&#9679;</FONT><BR></BR><B>+{remaining_tokens[place]}</B>>'
                penwidth = "2"
                fontsize = "12"
            elif place in missing_tokens:
                fillcolor = "lightcoral"
                label = f'<<B>-{missing_tokens[place]}</B>>'
                if place in initial_marking:
                    label = f'<<FONT POINT-SIZE="28">&#9679;</FONT><BR></BR><B>-{missing_tokens[place]}</B>>'
                penwidth = "2"
                fontsize = "12"

        return fillcolor, label, penwidth, fontsize, shape

    @staticmethod
    def display(graph: Digraph) -> None:
        display(Image(graph.render()))

    @staticmethod
    def save(graph: Digraph, file_name: str) -> None:
        graph.render(filename=file_name, format='png', cleanup=True)
