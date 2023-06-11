"""Implemeent Dijikstra algorithm to find the shortest path in an undirected weighted graph

Answer to the second programming assignment of  https://www.coursera.org/learn/algorithms-graphs-data-structures:
>>> python -m src.dijakstra -i data/dijkstra.txt
"""

import logging
import argparse
import heapq
from typing import Union, List, Tuple

from src.data_struct import Node, Graph

logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)


CORRECT_ANSWER = [2599,2610,2947,2052,2367,2399,2029,2442,2505,3068]

# to do: implement own heap, right now O(m*n)
class DijkstraFast:
    """O(m logn), using heap data structure"""

    def __init__(self, graph: Graph, undefined_dist: int = 1000000) -> None:
        self.graph = graph
        self.undefined_dist = undefined_dist
        self.distances = {} # store shortest-path distances to the key node
        self.prev = {}      # store previous node that pointed to key Node on the shortest path
        self.visited_nodes = set() # set of X:nodes that have already been processed
        self.min_heap = []
        self.name_to_node = {} # mapping between node name and the actual Node

        for node in self.graph:
            self.prev[node.name] = None
            self.name_to_node[node.name] = node
            self.distances[node.name] = undefined_dist
        self.graph._count_incoming()

    def run(self, source: Node) -> None:
        """Find the shortest path from source to each of the nodes in the graph"""
        self._construct_heap(source)
        while len(self.visited_nodes) != len(self.graph):
            dij_score, child = self._get_min_score_edge()
            parent = self.prev[child.name]
            log.debug(parent)
            if parent:
                assert self.name_to_node[parent] in self.visited_nodes
            log.debug(f'Edge with the min Dij. score {dij_score}: {parent} -> {child.name}')
            assert child not in self.visited_nodes
            self.visited_nodes.add(child)
            self.distances[child.name] = dij_score
            self._maintain_heap_invariant(dij_score, child)

    def _get_min_score_edge(self) -> Tuple[int, Node]:
        """Find the edge that has the minimum Dijkstra greedy score defined as
        score = A[v] + length[v, w]
        """
        score, child_name = heapq.heappop(self.min_heap)
        return score, self.name_to_node[child_name]

    def _maintain_heap_invariant(self, dij_score: int, head: Node) -> None:
        """Make sure that all children of the newly added Node has up-to-date scores in the heap

        Parameters
        ----------
        head: Node, is the newly added Node to the set of visited nodes
        dij_score: int, is the Dijkstra score for the head Node
        """
        for child in head.children:
            if child in self.visited_nodes:
                continue

            new_score = dij_score + head.children_weights[child.name]
            old_score, _ = self._delete_from_heap(child.name)
            assert old_score == self.distances.get(child.name, self.undefined_dist), \
              f'Adding {head.name}, old score for {child.name} deviates: {old_score}, {new_score}'
            if new_score < old_score:
                log.debug(f'Node: {child.name}: New score: {new_score}, old_score: {old_score}')
                log.debug(f"Pushing new entry into heap: {new_score, child.name}")
                heapq.heappush(self.min_heap, (new_score, child.name))
                self.prev[child.name] = head.name
                self.distances[child.name] = new_score
            else:
                heapq.heappush(self.min_heap, (old_score, child.name))

    def _construct_heap(self, source: Node) -> None:
        """Heap contains a tuple with 3 values, e.g. (1, '5', '7') means
        that the edge Node(5) -> Node(7) with Node(5) being in visited set X and
        Node(7) belonging to set unvisited set V-X has Dijkstra greedy score 1
        """
        for node in self.graph:
            if node == source:
                self.min_heap.append((0, source.name))
            else:
                self.min_heap.append((self.undefined_dist, node.name))
        heapq.heapify(self.min_heap)
        log.debug(len(self.min_heap))

    def _delete_from_heap(self, node_name_to_delete: Union[int, str]):
        """Linear amount of work due to heapify and list deletion"""
        for entry in self.min_heap:
            _, node_name = entry
            if node_name == node_name_to_delete:
                log.debug(f'Deleting {node_name_to_delete}')
                self.min_heap.remove(entry)
                heapq.heapify(self.min_heap)
                return entry

    def print_shortest_path(self, target_node_name) -> str:
        path = []
        previous_node = self.prev[target_node_name]
        while previous_node:
            path.append(previous_node)
            previous_node = self.prev[previous_node]
        if len(path) > 0:
            path.insert(0, target_node_name)
        return ' > '.join(str(x) for x in path[::-1])

class DijkstraBasic:
    """O(m * n), no heap data structure"""

    def __init__(self, graph: Graph, undefined_dist: int = 1000000) -> None:
        self.graph = graph
        self.undefined_dist = undefined_dist
        self.distances = {} # store shortest-path distances to the key node
        self.prev = {}      # store previous node that pointed to key Node on the shortest path
        self.visited_nodes = set() # set of X:nodes that have already been processed

        for node in self.graph:
            node.visited = False
            self.distances[node.name] = self.undefined_dist
            self.prev[node.name] = None

    def run(self, source: Node):
        """Find the shortest path from source to each of the nodes in the graph"""
        self.distances[source.name] = 0
        self.visited_nodes.add(source)
        while self.visited_nodes != len(self.graph):
            current_node, child = self._get_min_score_edge()
            log.debug('Current node:', current_node, 'child:', child)
            if not current_node:
                break
            self.visited_nodes.add(child)
            self.distances[child.name] = self.distances[current_node.name] + current_node.children_weights[child.name]
            self.prev[child.name] = current_node.name

    def _get_min_score_edge(self):
        """Find the edge that has the minimum Dijkstra greedy score defined as
        score = A[v] + length[v, w]
        """
        score = self.undefined_dist
        min_node, min_child = None, None
        for node in self.visited_nodes:
            node_score = self.distances[node.name]
            for child in node.children:
                if child in self.visited_nodes:
                    continue
                new_score = node_score + node.children_weights[child.name]
                if new_score < score:
                    score = new_score
                    min_node = node
                    min_child = child
        return min_node, min_child

    def print_shortest_path(self, target_node_name):
        path = []
        previous_node = self.prev[target_node_name]
        while previous_node:
            path.append(previous_node)
            previous_node = self.prev[previous_node]
        if len(path) > 0:
            path.insert(0, target_node_name)
        return ' > '.join(str(x) for x in path[::-1])

def graph_from_adj_list(adj_list: List[Tuple[int, List[List[int]]]]) -> Graph:
    """Given list of dependencies, generate a graph object to work with"""
    unique_nodes = {}
    for node_name, edges in adj_list:
        if not node_name in unique_nodes:
            unique_nodes[node_name] = Node(node_name)

        for (child_name, weight) in edges:
            if not child_name in unique_nodes:
                unique_nodes[child_name] = Node(child_name)
            unique_nodes[node_name].children.append(unique_nodes[child_name])
            unique_nodes[node_name].children_weights[child_name] = weight

    return Graph(nodes=list(unique_nodes.values()), weighted=True)

def parse_edge(line: str):
    row = line.split('\t')
    node, edges = row[0], row[1:-1]
    edges = [list(map(int, e.split(','))) for e in edges]
    return int(node), edges

def load_graph(file_path: str) -> str:
    with open(file_path, 'r') as file:
        for line in file:
            yield line.rstrip('\n')

if __name__ == '__main__':
    log.info('Loading the data...')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Path to .txt with graph in adjacency list format with weights')
    parser.add_argument('-d', '--destination', required=False, nargs="+", type=int,
                        default = [7,37,59,82,99,115,133,165,188,197],
                        help='Specify list of vertices to print the shortest distance to')
    args = parser.parse_args()
    raw_data = list(load_graph(args.input))
    adj_list = [parse_edge(edge) for edge in raw_data]
    graph = graph_from_adj_list(adj_list)
    #assert len(graph) == 200

    dj = DijkstraFast(graph=graph)
    dj.run(graph[0])

    if args.destination:
        log.info(f"Shortest paths from source to nodes: {','.join(map(str, args.destination))} is: ")
        answer = [dj.distances[t] for t in args.destination]
        #assert answer == CORRECT_ANSWER
        log.info(','.join(map(str, answer)))
