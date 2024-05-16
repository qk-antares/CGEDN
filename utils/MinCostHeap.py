import heapq
from itertools import count

class MinCostHeapWithCounter:
    def __init__(self):
        self.heap = []
        self.counter = count(start=-1, step=-1)
    
    def push(self, path, cost):
        # when the cost is the same, the latest path should be placed at heap top
        item = (cost, next(self.counter), path)
        heapq.heappush(self.heap, item)
    
    def pop(self):
        if self.heap:
            # pop the path and cost
            return heapq.heappop(self.heap)
        else:
            raise IndexError("pop from an empty heap")
        
    def size(self):
        return len(self.heap)

    def resize(self, beamsize):
        if beamsize >= len(self.heap):
            return
        elif beamsize <= 0:
            self.heap = []
        else:
            self.heap = heapq.nsmallest(beamsize, self.heap)

# 创建一个 MinCostHeap 实例
min_cost_heap = MinCostHeapWithCounter()

# 添加一些元素
min_cost_heap.push("A", 5)
min_cost_heap.push("B", 3)
min_cost_heap.push("C", 7)
min_cost_heap.push("D", 2)
min_cost_heap.push("E", 6)
min_cost_heap.push('F', 2)

# 打印堆中的元素
print("当前堆中的元素：", min_cost_heap.heap)

# 弹出最小cost的元素
print("弹出的最小cost的元素：", min_cost_heap.pop())
print("弹出的最小cost的元素：", min_cost_heap.pop())

# 打印堆中的元素
print("当前堆中的元素：", min_cost_heap.heap)

# 调整堆的大小
min_cost_heap.resize(2)

# 打印堆中的元素
print("调整大小后的堆中的元素：", min_cost_heap.heap)