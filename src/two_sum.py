"""Variation of two-sum problem where you have a large number of target values in a given range.

Answer to the fourth programming assignment of  https://www.coursera.org/learn/algorithms-graphs-data-structures:
>>> python -m src.two_sum -i data/2sum.txt --min -10000 --max 10000
"""
import argparse
import logging
from typing import List
from bisect import bisect_left

logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)


class TwoSumModLogn:
    """~O(n * log n) due to binary search"""
    def __init__(self, nums: List[int]):
        self.nums = sorted(nums)
        self.total = len(nums)
        self.found_targets = set()

    def count_num_targets(self, target_from: int, target_till: int) -> int:
        for x in self.nums:
            y_min = target_from - x
            y_max = target_till - x
            idx_min = bisect_left(self.nums, y_min)
            idx_max = bisect_left(self.nums, y_max)
            for y in self.nums[idx_min:idx_max]:
                self.found_targets.add(x + y)
        return len(self.found_targets)

def get_numbers(file_path: str) -> int:
    with open(file_path, 'r') as f:
        for line in f:
            yield int(line.strip('\n'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Path to .txt with the list of numbers')
    parser.add_argument('--min', required=True, type=int,
                        help='Minumum target range')
    parser.add_argument('--max', required=True, type=int,
                        help='Maximum target range')
    args = parser.parse_args()
    log.info(f'Loading data from {args.input}...')

    task = TwoSumModLogn([n for n in get_numbers(args.input)])
    answer = task.count_num_targets(args.min, args.max)
    log.info(f'Answer to programming assignment #4: {answer}')
