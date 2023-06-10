"""Implementation for later reusal"""

import logging
from collections import defaultdict
from typing import Union, List

from graphviz import Digraph

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

class Node:
    def __init__(self, name: Union[str, int], children: List[int] = None):
        self.name = name
        if children:
            self.children = children
        else:
            self.children = []
        self.inbound = 0
        self.visited = False
        self.children_weights = {}

    def __repr__(self):
        node_name = [f'Node {self.name} with {self.inbound} inbound edges']
        #if len(self.children) > 0:
            #node_name.append(f' and with children {", ".join(str(x) for x in self.children)}.')
        return ''.join(node_name)

class Graph:
    def __init__(self, nodes: List[Node],  weighted: bool = False):
        self.nodes = nodes
        self.weighted = weighted

    def __repr__(self):
        self._clean_incoming()
        self._count_incoming()
        if len(self.nodes) > 1000:
            return f"Graph with {len(self.nodes)}"
        return '\n'.join(str(node) for node in self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, key: int):
        return self.nodes[key]

    def _count_incoming(self):
        for node in self.nodes:
            for child in node.children:
                child.inbound += 1

    def _clean_incoming(self):
        for node in self.nodes:
            node.inbound = 0
            for child in node.children:
                child.inbound = 0

    def _clean_visits(self) -> None:
        for node in self.nodes:
            node.visited = False

    def reverse(self) -> None:
        """Reverse all the edges in the graph aka tranpose"""
        reversed_children = defaultdict(list)
        for node in self.nodes:
            for child in node.children:
                reversed_children[child.name].append(node)
        for node in self.nodes:
            node.children = reversed_children[node.name]
        del(reversed_children)

    def plot(self):
        if len(self.nodes) > 100:
            print("Unsafe to viz more than 100 nodes")
            return
        from graphviz import Digraph
        g = Digraph(comment='My Graph', format='png')
        g.attr(rankdir='LR')
        g.attr('node', shape='circle')

        for node in self.nodes:
            g.node(str(node.name))
            for child in node.children:
                if self.weighted:
                    g.edge(str(node.name), str(child.name), label=str(node.children_weights[child.name]))
                else:
                    g.edge(str(node.name), str(child.name))
        return g
