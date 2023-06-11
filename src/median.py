"""Compute median on the stream of data

Answer to the third programming assignment of  https://www.coursera.org/learn/algorithms-graphs-data-structures:
>>> ppython -m src.median -i data/hw_median.txt
"""

import heapq
import logging
import argparse
import heapq

logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

class MedianFinder:

    def __init__(self):
        self.heap_low = []
        self.heap_high = []
        heapq.heapify(self.heap_low)
        heapq.heapify(self.heap_high)

    def addNum(self, num: int) -> None:
        if len(self.heap_low) == 0:
            heapq.heappush(self.heap_low, -num)
            return
        if len(self.heap_high) == 0:
            heapq.heappush(self.heap_high, num)

            # make sure that low heap is low after first 2 values arrived
            if -self.heap_low[0] > self.heap_high[0]:
                num_low = -heapq.heappop(self.heap_low)
                num_high = heapq.heappop(self.heap_high)

                heapq.heappush(self.heap_low, -num_high)
                heapq.heappush(self.heap_high, num_low)
            return

        self._push(num)
        if abs(len(self.heap_low) - len(self.heap_high)) > 1:
            self._rebalance()

    def _push(self, num):
        """Pick the heap to push the number to"""
        if num > -self.heap_low[0]:
            heapq.heappush(self.heap_high, num)
        else:
            heapq.heappush(self.heap_low, -num)

    def _rebalance(self):
        """Make sure that heap_low and heap_high have similar number of items"""
        if len(self.heap_low) > len(self.heap_high):
            num = -heapq.heappop(self.heap_low)
            heapq.heappush(self.heap_high, num)
        else:
            num = heapq.heappop(self.heap_high)
            heapq.heappush(self.heap_low, -num)

    def findMedian(self) -> float:
        if len(self.heap_low) > len(self.heap_high):
            return -self.heap_low[0]
        elif len(self.heap_low) < len(self.heap_high):
            return self.heap_high[0]
        else:
            # median of k even numbers is the (k/2)th lowest number, not the average
            #return (-self.heap_low[0] + self.heap_high[0]) / 2
            return -self.heap_low[0]

def get_numbers(file_path: str) -> int:
    with open(file_path, 'r') as f:
        for line in f:
            yield int(line.strip('\n'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Path to .txt with the list of numbers')
    args = parser.parse_args()
    log.info(f'Loading data from {args.input}...')

    mf = MedianFinder()
    medians = []
    for num in get_numbers(args.input):
        mf.addNum(num)
        medians.append(mf.findMedian())

    answer = sum(medians) % 10000
    log.info(f'Answer to programming assignment #3: {answer}')
