
import os

from graphviz import Digraph
from enum import Enum


class Relations(Enum):
    LOOP = 'O'# LOOP = 'loop'   # 
    SEQUENCE =  '->'   #seq =  '->'
    CHOICE = 'X'   #choice = 'X'
    TAU = 'tau'   # tau = 'tau'
    PARALLEL = '||'  #parallel = '||'
 
class Visualisation:
    def visualize_process_tree(self, tree):
        # Initialize a Graphviz Digraph object
        dot = Digraph()
        
        # Recursive function to add nodes and edges to the graph
        # node_id to keep track of hierarchy level of the tree
        def add_nodes_edges(tree, parent=None, node_id=0):
            if isinstance(tree, tuple):
                node_label = tree[0]
                children = tree[1]
            else:
                node_label = tree
                children = []
            
            current_id = str(node_id)
          
            match node_label: 
                case Relations.LOOP.value:
                    image_path = os.path.join(os.path.dirname(__file__), 'images/loop_arrow.png')
                    dot.node(current_id, image=image_path, shape="ellipse", fixedsize="true", label="", width="0.6", height="0.7", penwidth = "2")  
                case Relations.PARALLEL.value: 
                    image_path = os.path.join(os.path.dirname(__file__), 'images/and.png')  
                    dot.node(current_id, image=image_path, shape="ellipse", label="", fixedsize="true", width="0.6", height="0.7", penwidth = "2")
                case Relations.SEQUENCE.value:
                    image_path = os.path.join(os.path.dirname(__file__), 'images/seq_arrow.png') 
                    dot.node(current_id, image=image_path, shape="ellipse", label="", fixedsize="true", width="0.6", height="0.7", penwidth = "2")
                case Relations.CHOICE.value:
                    image_path = os.path.join(os.path.dirname(__file__), 'images/x.png') 
                    dot.node(current_id, image=image_path, shape="ellipse", label="", fixedsize="true", width="0.6", height="0.7", penwidth = "2")
                case Relations.TAU.value:
                    image_path = os.path.join(os.path.dirname(__file__), 'images/tau.png')
                    dot.node(current_id, image=image_path, shape="square", label="", fixedsize="true", width="0.6", height="0.7", penwidth = "2",imagescale = "true")
                case None:
                    image_path = os.path.join(os.path.dirname(__file__), 'images/bomb.png')
                    dot.node(current_id, image=image_path, imagescale = "true", shape="square", label="", fixedsize="true", width="0.6", height="0.7", penwidth = "2",fontname="Arial", fontcolor="black")
                case _:   
                    dot.node(current_id, node_label,  shape="square", fixedsize="true", width="0.6", height="0.7",  fontsize="30", fontname="Arial", fontcolor="black" ) #shape="circle"
             
         
            if parent is not None:
                dot.edge(parent, current_id)
            
            for i, child in enumerate(children):
                add_nodes_edges(child, current_id, node_id * 10 + i + 1)
        
        # Start the recursion with the root of the tree
        add_nodes_edges(tree)
        
      
      
        dot.render('tree', format='png', view=True)
    

# Define the tree structure
# example trees
#tree = ('->', ['a', ('X', ['tau', 'b']), ('X', ['tau', 'c']), 'd'])

#tree = [('->', ['a', 'b', 'c']), ('->', [('||', [None, None]), 'k']), ('->', [None, 'g', 'h'])]

#tree  = ('->', ['a', ('X', ['e', ('||', ['b', 'c'])]), 'd'])


#tree = ('->', ['a', ('X', ['tau', 'b']), ('X', ['tau', 'c']), 'd'])

#tree = ('loop', ['a', ('X', ['tau', 'b']), ('X', ['tau', 'c']), 'd'])

#tree = ('O', [('->', ['a', 'b', 'c']), ('->', [('||', [None, None]), 'k']), ('->', [None, 'g', 'h'])])

#tree = ('O', [('->', ['a', 'b', 'c']), ('->', [('O', ['tau', 'i', 'j']), 'k']), ('->', [('O', ['tau', 'd', 'f', 'e']), 'g', 'h'])])


#tree = ("->", ['a',            ("loop", [("->", ( "and", [( "X", ['b','c']  )      ,"d" ])    )   , "e"]  , "f" )       ,                 ('X', ['g', 'h'] )         ])


#visobject = Visualisation()
#visobject.visualize_process_tree(tree)
