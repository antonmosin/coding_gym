"""Find strongly connected components in a graph

Answer to the first programming assignment of  https://www.coursera.org/learn/algorithms-graphs-data-structures:
>>> python -m src.scc -i data/scc.txt
"""

import logging
import argparse
from collections import defaultdict
from typing import Union, List, Dict, Tuple
import pandas as pd

from src.data_struct import Node, Graph

logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

class Scc:
    def __init__(self, graph: Graph, iterative: bool = True) -> None:
        self.graph = graph
        self.leaders = {}
        self.finish_order = []
        self.scc = defaultdict(list)
        self.iterative = iterative

    def find_scc(self) -> Dict[Union[str, int], Union[str, int]]:
        self.graph.reverse()
        log.info('Initiate first loop')
        self._dfs_loop()
        log.info('clean visits count')
        self.graph._clean_visits()
        log.info("Reverse graph back to original")
        self.graph.reverse()
        self.graph.nodes = self.finish_order[::-1]
        log.info('Initiate second loop')
        self._dfs_loop()
        self.finish_order = self.finish_order[:len(self.graph)]

        for node, leader_node in self.leaders.items():
            self.scc[leader_node].append(node)
        return self.scc

    def _dfs_loop(self) -> None:
        """Initiate depth-first-search on the entire graph"""
        for node in self.graph:
            if not node.visited:
                if self.iterative:
                    self._dfs_iterative(node)
                else:
                    self._dfs(node)

    def _dfs_iterative(self, node: Node) -> None:
        """Iterative implementation of depth-first-search that avoids deep recursions"""
        stack = []
        stack.append(node)
        finish_times_rev = []

        while stack:
            x = stack.pop()
            x.visited = True
            self.leaders[x.name] = node.name
            finish_times_rev.append(x)

            for child in x.children:
                if not child.visited:
                    stack.append(child)

        self.finish_order.extend(finish_times_rev[::-1])

    def _dfs(self, node: Node) -> None:
        """Recursive DFS that will fail on very large graph"""
        node.visited = True
        # running dfs on the original graph to obtain leader nodes
        self.leaders[node.name] = node.name
        for child in node.children:
            if not child.visited:
                #print(child)
                self._dfs(child, node)

        # running dfs on the reversed graph to obtain finishing times
        self.finish_order.append(node)

def load_graph(file_path: str) -> str:
    """Yield one line from the input file"""
    with open(file_path, 'r') as file:
        for line in file:
            yield line.rstrip('\n')

def parse_edge(line: str) -> Tuple[int, int]:
    """Interpret the line as parent node and child """
    node, child = line.split()
    return int(node), int(child)

def graph_from_dependencies(dependencies: List[List[int]]) -> Graph:
    """Given list of dependencies, generate a graph object to work with"""
    unique_nodes = {}
    for node_name, child_name in dependencies:
        if not node_name in unique_nodes:
            unique_nodes[node_name] = Node(node_name)
        if child_name:
            if not child_name in unique_nodes:
                unique_nodes[child_name] = Node(child_name)
            unique_nodes[node_name].children.append(unique_nodes[child_name])
    return Graph(nodes=list(unique_nodes.values()))


if __name__ == '__main__':
    log.info('Loading the data...')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Path to .txt with graph in adjacency list format')
    parser.add_argument('--top-n', dest='top_n', default=5, type=int, required=False,
                        help='Number of largest strongly components to print')
    args = parser.parse_args()

    deps = [parse_edge(edge) for edge in load_graph(args.input)]
    g_task = graph_from_dependencies(deps)
    #assert len(g_task) == 875714 # graph size from https://www.coursera.org/learn/algorithms-graphs-data-structures

    result = Scc(g_task).find_scc()
    # find top 5 components
    top = pd.Series(result).apply(len).sort_values(ascending=False)[:args.top_n].values
    answer = ','.join((str(x) for x in top))
    log.info(f'Size of the {args.top_n} largest strongly connected components: {answer}')
