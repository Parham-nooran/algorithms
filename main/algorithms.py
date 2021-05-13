import math
import random
from enum import Enum

import numpy as np


def mergeSort(input):
    if len(input) <= 1:
        return input
    return merge(mergeSort(input[0:int(len(input) / 2)]), mergeSort(input[int(len(input) / 2):len(input)]))


def merge(a, b):
    j = 0
    k = 0
    result = []
    while j < len(a) or k < len(b):
        if j < len(a) and k < len(b):
            if a[j] < b[k]:
                result.append(a[j])
                j += 1
            else:
                result.append(b[k])
                k += 1
        elif j < len(a):
            result.append(a[j])
            j += 1
        else:
            result.append(b[k])
            k += 1
    return result


def binarySearch(input, end):
    print(input)
    if len(input) < 1:
        return -1
    mid = int(len(input) / 2)
    if input[mid] == end:
        return mid
    elif end < input[mid]:
        return binarySearch(input[:mid], end)
    else:
        return int(len(input) / 2) + binarySearch(input[mid:], end)


def bigIntMult(u, v):
    n = max(math.ceil(math.log(u + 0.1, 10)), math.ceil(math.log(v + 0.1, 10)))
    if n < 3:
        return u * v
    else:
        m = math.floor(n / 2)
        x = int(math.floor(u / 10 ** m))
        y = u % 10 ** m
        w = int(math.floor(v / 10 ** m))
        z = v % 10 ** m
        p = bigIntMult(x, w)
        q = bigIntMult((x + y), (z + w))
        r = bigIntMult(y, z)
        return (10 ** (2 * m)) * p + (q - p - r) * (10 ** m) + r


def strassen(a, b):
    if len(a) == 1 and len(b) == 1:
        return [a[0] * b[0]]
    a11 = a[:int(len(a) / 2), :int(len(a[0]) / 2)]
    a12 = a[:int(len(a) / 2), int(len(a[0]) / 2):]
    a21 = a[int(len(a) / 2):, :int(len(a[0]) / 2)]
    a22 = a[int(len(a) / 2):, int(len(a[0]) / 2):]
    b11 = b[:int(len(b) / 2), :int(len(b[0]) / 2)]
    b12 = b[:int(len(b) / 2), int(len(b[0]) / 2):]
    b21 = b[int(len(b) / 2):, :int(len(b[0]) / 2)]
    b22 = b[int(len(b) / 2):, int(len(b[0]) / 2):]
    m1 = strassen((a11 + a22), (b11 + b22))
    m2 = strassen(a21 + a22, b11)
    m3 = strassen(a11, b12 - b22)
    m4 = strassen(a22, b21 - b11)
    m5 = strassen(a11 + a12, b22)
    m6 = strassen(a21 - a11, b11 + b12)
    m7 = strassen(a12 - a22, b21 + b22)
    m2 = np.array(m2)
    m3 = np.array(m3)
    m4 = np.array(m4)
    m5 = np.array(m5)
    result = np.empty((len(m1) + len(m2), len(m3) + len(m1)), type(m1[0][0]))
    result.fill(0)
    result[:int(len(m1)), :int(len(m1[0]))] = m1 + m4 - m5 + m7
    result[:int(len(m1)), int(len(m1[0])):] = m3 + m5
    result[int(len(m1)):, :int(len(m1[0]))] = m2 + m4
    result[int(len(m1)):, int(len(m1[0])):] = m1 + m3 - m2 + m6
    return result


def turnSquare(a, l):
    n = max(len(a), len(a[0]), l)
    n = int(math.pow(2, math.ceil(math.log(n, 2))))
    zero = np.zeros((n, n), type(a[0][0]))
    zero[:len(a), :len(a[0])] = a
    return zero


def bino_rec(n, k):
    if k == 0 or n == k:
        return 1
    return bino(n - 1, k - 1) + bino(n - 1, k)


def bino(n, k):
    result = np.zeros((n + 1, k + 1))
    for i in range(0, n + 1):
        for j in range(0, min(k, i) + 1):
            if j == 0 or i == j:
                result[i][j] = 1
            else:
                result[i][j] = result[i - 1][j - 1] + result[i - 1][j]
    print(result)
    return result[n][k]


def fibo(n):
    result = [0] * n
    for i in range(0, n):
        if i < 2:
            result[i] = 1
        else:
            result[i] = result[i - 1] + result[i - 2]
    print(result)
    return result[n - 1]


def shortestPath(w):
    d = w
    n = len(w)
    p = np.empty((n, n), int)
    p.fill(-1)
    o = np.ones((n, n))
    for k in range(0, n):
        for i in range(0, n):
            for j in range(0, n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
                    p[i][j] = k
    print(p + o)
    print(d)


def minMult(d):
    n = len(d) - 1
    m = np.zeros((n, n))
    p = np.zeros((n, n))
    for diagonal in range(0, n):
        for i in range(0, n - diagonal):
            j = i + diagonal
            if i == j:
                m[i][j] = 0
            else:
                m[i][j] = math.inf
                for k in range(i, j):
                    if m[i][k] + m[k + 1][j] + d[i] * d[j] * d[k] < m[i][j]:
                        m[i][j] = m[i][k] + m[k + 1][j] + d[i] * d[j + 1] * d[k + 1]
                        print(str(i) + str(j) + str(k) + " : " + str(m[i][j]))
                        p[i][j] = k

    print("p is : ")
    print(p)
    print("m is : ")
    print(m)


def stoogeSort(a, i, j):
    n = j - i + 1
    if n <= 1:
        return a[0]
    if n == 2:
        if a[i] > a[j]:
            swap(a, i, j)
    else:
        m = math.floor(n / 3)
        stoogeSort(a, i, j - m)
        stoogeSort(a, i + m, j)
        stoogeSort(a, i, j - m)


# def maxSubArrayNaive(a):
#     maximum = a[0]
#     s = 0
#     e = 0
#     for i in range(0, len(a)):
#         sum = 0
#         for j in range(i, len(a)):
#             sum += a[j]
#             if sum > maximum:
#                 maximum = sum
#                 s = i
#                 e = j
#     return Result(maximum, s, e)


class Result:
    def __init__(self, maximum, start, end):
        self.maximum = maximum
        self.start = start
        self.end = end

    def __repr__(self):
        return "Maximum sum is : " + str(self.maximum) + " start index is : " + \
               str(self.start) + " end index is : " + str(self.end)


def swap(a, i, j):
    temp = a[i]
    a[i] = a[j]
    a[j] = temp


def sort(input):
    for i in range(0, len(input)):
        for j in range(0, len(input)):
            if input[i] < input[j]:
                swap(input, i, j)


class Color(Enum):
    RED = 1
    BLACK = 2

    def __str__(self):
        return self.name


def partition(a, p):
    L = []
    R = []
    for i in range(0, len(a)):
        if i == p:
            continue
        if a[i] < a[p]:
            L.append(a[i])
        else:
            R.append(a[i])
    return L, a[p], R


def chooseRandomPivot(n):
    return int(random.random() * n)


# def select(a, k):
#     # print(f"a : {a} k : {k}")
#     if len(a) < 10:
#         # print("end")
#         return mergeSort(a)[k-1]
#     p = chooseRandomPivot(len(a))
#     L, mid, R = partition(a, p)
#     # print(f"p : {p}")
#     if len(L) > k:
#         # print(f"L : {L}", k)
#         return select(L, k)
#     elif len(L) == k:
#         return mid
#     else:
#         # print(f"R : {R}", k - len(L) - 1)
#         return select(R, k-len(L)-1)
def select(a, k):
    if len(a) < 50:
        return mergeSort(a)[k - 1]
    p = chooseRandomPivot(len(a))
    L, mid, R = partition(a, p)
    if len(L) == k:
        return mid
    elif len(L) > k:
        return select(L, k)
    else:
        return select(R, k - len(L) - 1)


def isSortedAsc(a):
    for i in range(0, len(a) - 1):
        if a[i] > a[i + 1]:
            return False
    return True


def isSortedDesc(a):
    for i in range(0, len(a) - 1):
        if a[i] < a[i + 1]:
            return False
    return True


def bogoSort(a):
    while not isSortedAsc(a):
        random.shuffle(a)


def quickSort(a):
    if len(a) < 3:
        return mergeSort(a)
    L, mid, R = partition(a, chooseRandomPivot(len(a)))
    return quickSort(L) + [mid] + quickSort(R)


def random1dArray(n=5, d=3):
    test = []
    for j in range(0, n):
        test.append(int(random.random() * 10 ** d))
    return test


def polarRandom1dArray(n=5, d=3):
    test = []
    for j in range(0, n):
        test.append(int(random.random() * 10 ** d) - int(random.random() * 10 ** d))
    return test


def countingSort(a):
    M = max(a)
    buckets = [-1 for i in range(0, M + 1)]
    for i in range(0, len(a)):
        buckets[a[i]] = a[i]
    result = []
    for i in range(0, M + 1):
        if buckets[i] != -1:
            result += [buckets[i]]
    return result


def radixSort(a, d=3, r=10):
    n = len(a)
    for j in range(0, d):
        result = [[] for k in range(0, r)]
        for i in range(0, n):
            result[int((a[i] % r ** (j + 1) - a[i] % r ** j) / r ** j)].append(a[i])
        a = []
        for m in range(0, r):
            a = a + result[m]
    return a


def search(head, key):
    if head.key == key:
        return head
    elif key < head.key:
        if head.left != None:
            return search(head.left, key)
        else:
            return None
    else:
        if head.right != None:
            return search(head.right, key)
        else:
            return None


def insert(head, key):
    if head.key == key:
        return False
    if key > head.key:
        if head.right is not None:
            return insert(head.right, key)
        head.right = Node(key, None, head)
        return True
    if head.left is not None:
        return insert(head.left, key)
    head.left = Node(key, None, head)
    return True


def deleteFromTree(head, key):
    temp = search(head, key)
    if temp is not None:
        deleteNode(temp)


def deleteNode(node):
    if node.left is None or node.right is None:
        print("Something is None")
        if node.right is not None:
            node.parent.set_right_child(node.right)
            node.delete()
        elif node.left is not None:
            node.parent.set_left_child(node.right)
        else:
            node.delete()
    else:
        print("Nothing is None")
        print(node.key)
        temp = findImmediateSuccessor(node)
        if temp is not None:
            node.set_key(temp.key)
            deleteNode(temp)


def findImmediateSuccessor(node):
    if node.right is None:
        return None
    return findSmallestLeft(node.right)


def findSmallestLeft(node):
    if node.left is None:
        return node
    return findSmallestLeft(node.left)


def traversePostOrder(head):
    if head.left is not None:
        traversePostOrder(head.left)
    if head.right is not None:
        traversePostOrder(head.right)
    print(head)


def traverseInOrder(head):
    if head.left is not None:
        traverseInOrder(head.left)
    print(head)
    if head.right is not None:
        traverseInOrder(head.right)


def traversePreOrder(head):
    print(head)
    if head.left is not None:
        traversePreOrder(head.left)
    if head.right is not None:
        traversePreOrder(head.right)


"""
    returns the found subArray and the result of the summation
"""


def maxSubArray(A):
    n = len(A)
    current_max = global_max = A[0]
    startCur = endCur = 0
    start = end = 0
    for i in range(1, n):
        if A[i] > current_max + A[i]:
            current_max = A[i]
            startCur = endCur = i
        else:
            current_max += A[i]
            endCur = i
        if current_max > global_max:
            global_max = current_max
            start = startCur
            end = endCur
    return global_max, A[start:end + 1]


class Vertex:
    def __init__(self, startTime, finishTime, state, neighbors, key=None):
        self.key = key
        self.startTime = startTime
        self.finishTime = finishTime
        self.state = state
        self.neighbors = neighbors


class DVertex(Vertex):
    def __init__(self, toNeighbors, fromNeighbors, startTime, finishTime, state, neighbors):
        super().__init__(startTime, finishTime, state, neighbors)
        self.toNeighbors = toNeighbors
        self.fromNeighbors = fromNeighbors


class State(Enum):
    UNVISITED = 0
    IN_PROGRESS = 1
    ALL_DONE = 2


def depthFirstSearch(w, currentTime):
    w.startTime = currentTime
    currentTime += 1
    w.state = State.IN_PROGRESS
    for v in w.neighbors:
        if v.state == State.UNVISITED:
            currentTime = depthFirstSearch(v, currentTime)
            currentTime += 1
    w.finishTime = currentTime
    w.state = State.ALL_DONE
    return currentTime


def inplcMergeSort(a, start, end):
    if end <= start:
        return a
    mid = (start + end) // 2
    inplcMergeSort(a, start, mid)
    inplcMergeSort(a, mid + 1, end)
    # print(test[start:mid])
    # print(test[mid:end])
    inplcMerge(a, start, end)
    # print(a[start:end])


def inplcMerge(a, start, end):
    mid = (start + end) // 2
    start2 = mid + 1
    while start <= mid and start2 <= end:
        if a[start] <= a[start2]:
            start += 1
        else:
            val = a[start2]
            index = start2
            while index > start:
                a[index] = a[index - 1]
                index -= 1
            a[start] = val
            start2 += 1
            start += 1
            mid += 1


def nMergeSort(a, low, high):
    if high - low <= 1:
        return a[low:high]
    mid = (low + high) // 2
    return merge(nMergeSort(a, low, mid // 2), nMergeSort(a, mid // 2 + 1, mid))


def breadthFirstSearch():
    pass


def insertionSort(a):
    if a[0] > a[1]:
        a[1], a[0] = a[0], a[1]
    tail = 1
    for head in range(2, len(a)):
        if a[head] < a[tail]:
            val = a[head]
            a[head] = a[tail]
            point = tail
            while val < a[point] and point >= 0:
                a[point] = a[point - 1]
                point -= 1
            a[point + 1] = val
        tail += 1


def selectionSort(a):
    for start in range(0, len(a)):
        minimum = a[start]
        minI = start
        for i in range(start, len(a)):
            if a[i] < minimum:
                minimum = a[i]
                minI = i
        a[start], a[minI] = a[minI], a[start]


class Node:
    def __init__(self, key=0, color=None, parent=None, left=None, right=None):
        self.left = left
        self.right = right
        self.key = key
        self.parent = parent
        self.color = color

    def delete(self):
        if self.parent.left is self:
            self.parent.left = None
        elif self.parent.right is self:
            self.parent.right = None
        self.parent = None

    def __repr__(self):
        return f"Key : {self.key} + Color : {self.color}"

    def print_instance_name(self):
        print(self.__class__.__name__)

    # def set_color(self, color):
    #     self.color = color
    #
    # def set_parent(self, parent):
    #     self.parent = parent
    #
    # def set_key(self, value):
    #     self.key = value

    # def set_children(self, left=None, right=None):
    #     if self.color == Color.RED and (left.color == Color.RED or right.color == Color.RED):
    #         raise TypeError("Red nodes can not have red children")
    #     else:
    #         self.left = left
    #         self.right = right

    # def set_left_child(self, left):
    #     self.left = left
    #
    # def set_right_child(self, right):
    #     self.right = right


def RBTreeInsert(head, value):
    if value < head.key:
        if head.left is not None:
            RBTreeInsert(head.left, value)
        else:
            head.left = Node(value, head, None, None, Color.RED if head.Color == Color.BLACK else Color.BLACK)
    elif value > head.key:
        if head.right is not None:
            RBTreeInsert()
    else:
        pass


def karatsubaMultiplication(x, y, n):
    if n == 1:
        return x * y
    a = x // 10 ** (n // 2)
    b = x % 10 ** (n // 2)
    c = y // 10 ** (n // 2)
    d = y % 10 ** (n // 2)
    ac = karatsubaMultiplication(a, c, n // 2)
    bd = karatsubaMultiplication(b, d, n // 2)
    p = a + b
    q = c + d
    pq = karatsubaMultiplication(p, q, n // 2)
    return 10 ** (n) * ac + 10 ** (n // 2) * (pq - ac - bd) + bd


def insertionSortDecreasing(a):
    for i in range(1, len(a)):
        if a[i] > a[i - 1]:
            value = a[i]
            index = i
            while index > 0 and value > a[index - 1]:
                a[index] = a[index - 1]
                index -= 1
            a[index] = value


def union(a, b):
    result = []
    aIndex = 0
    bIndex = 0
    while aIndex < len(a) and bIndex < len(b):
        if len(result) > 0:
            if a[aIndex] == result[len(result) - 1]:
                aIndex += 1
                continue
            if b[bIndex] == result[len(result) - 1]:
                bIndex += 1
                continue
        if a[aIndex] <= b[bIndex]:
            result.append(a[aIndex])
        elif a[aIndex] > b[bIndex]:
            result.append(b[bIndex])
    while aIndex < len(a):
        if len(result) > 0 and a[aIndex] == result[len(result) - 1]:
            aIndex += 1
            continue
        result.append(a[aIndex])
        aIndex += 1
    while bIndex < len(b):
        if len(result) > 0 and b[bIndex] == result[len(result) - 1]:
            bIndex += 1
            continue
        result.append(b[bIndex])
        bIndex += 1
    return result


def intersection(a, b):
    aIndex = 0
    bIndex = 0
    result = []
    while aIndex < len(a) and bIndex < len(b):
        while aIndex < len(a) and a[aIndex] < b[bIndex]:
            aIndex += 1
        if aIndex >= len(a):
            break
        while bIndex < len(b) and b[bIndex] < a[aIndex]:
            bIndex += 1
        if bIndex >= len(b):
            break
        if a[aIndex] == b[bIndex]:
            result.append(a[aIndex])
            while aIndex < len(a) and a[aIndex] == result[len(result) - 1]:
                aIndex += 1
            while bIndex < len(b) and b[bIndex] == result[len(result) - 1]:
                bIndex += 1
    return result


def inplacePartition(a, p):
    i = 0
    while i < p:
        if a[i] > a[p]:
            val = a[i]
            for j in range(i, p):
                a[j] = a[j + 1]
            a[p] = val
            p -= 1
        else:
            i += 1
        print(a)
    i = p + 1
    while i < len(a):
        if a[i] < a[p]:
            val = a[i]
            for j in range(i, p, -1):
                a[j] = a[j - 1]
            a[p] = val
            p += 1
        else:
            i += 1
        print(a)


def findRightest(head):
    if head.right is None:
        return head
    return findRightest(head.right)


def B(x):
    if x is None:
        return 0
    if x.left is not None:
        return B(x.left) + (1 if x.color is Color.BLACK else 0)
    if x.right is not None:
        return B(x.right) + (1 if x.color is Color.BLACK else 0)
    return 1 if x.color is Color.BLACK else 0


def changeColorDownward(head, color):
    if head is None or head.color == color:
        return True
    head.color = color
    changeColorDownward(head.left, color.RED if color is Color.BLACK else Color.RED)
    changeColorDownward(head.right, color.RED if color is Color.BLACK else Color.RED)


def changeColor(head, color):
    # print("Change ", head, "to ", color)
    if head is None or head.color == color:
        return True
    if head.parent is None and color is Color.BLACK:
        head.color = color
        return True
    if head.parent is None and color is Color.RED:
        temp = findImmediateSuccessor(head)
        if temp is None:
            head.parent = head.left
            findRightest(head.left).right = head
            global A
            A = head.left
            A.parent = None
            A.right.left = None
            changeColorDownward(head.left, Color.RED if A.right.color is Color.BLACK else Color.BLACK)
            changeColorDownward(head.right, Color.RED if A.right.color is Color.BLACK else Color.BLACK)
        else:
            temp.left = head
            head.parent = temp
            A = head.right
            A.parent = None
            head.right = None
        changeColor(head, color)
        changeColorDownward(head.left, Color.RED if head.color is Color.BLACK else Color.BLACK)
        changeColorDownward(head.right, Color.RED if head.color is Color.BLACK else Color.BLACK)
    if head.parent.color is Color.RED and color is Color.RED:
        changeColor(head.parent, Color.BLACK)
    changeColor(head.parent, Color.RED if color is Color.BLACK else Color.BLACK)
    head.color = color
    return True


def RBInsert(head, value):
    if head is None or head.key == value:
        return False
    if value < head.key:
        if head.left is not None:
            return RBInsert(head.left, value)

        if B(head.left) < B(head.right):
            head.left = Node(value, Color.BLACK, head)
            return True
        if head.color is not Color.BLACK:
            changeColor(head, Color.BLACK)
            changeColor(head.right, Color.RED)
        head.left = Node(value, Color.RED if \
            # head.color is not Color.RED and \
            B(head.left) >= B(head.right) else Color.BLACK, head)
        return True

    if head.right is not None:
        return RBInsert(head.right, value)
    if B(head.left) > B(head.right):
        head.right = Node(value, Color.BLACK, head)
        return True
    if head.color is not Color.BLACK:
        changeColor(head, Color.BLACK)
        changeColor(head.left, Color.RED)
    head.right = Node(value, Color.RED if \
        # head.color is not Color.RED and \
        B(head.left) <= B(head.right) else Color.BLACK, head)
    return True


if __name__ == "__main__":
    # test = random1dArray(20)
    # print(test)
    # print(countingSort(test))
    # a = [[4, 2, 6, 7], [6, 7, 8, 1], [4, 5, 4, 2]]
    # b = [[4, 6], [6, 1]]
    # a = turnSquare(a, len(b))
    # b = turnSquare(b, len(a))
    # if len(a) != len(b):
    #     exit(-1)
    # print(strassen(a, b))
    # print(fibo(15))
    # given = [[0, 1, math.inf, 1, 5], [9, 0, 3, 2, math.inf], [math.inf, math.inf, 0, 4, math.inf],
    #          [math.inf, math.inf, 2, 0, 3], [3, math.inf, math.inf, math.inf, 0]]
    # shortestPath(given)
    # for i in range(0, 100):
    #     test = random1dArray(10)
    #     # print(test)
    #     insertionSortDecreasing(test)
    #     # print(test)
    #     # print(isSortedDesc(test))
    #     if not isSortedDesc(test):
    #         print(f"{i}", test)
    # a = [1, 3, 4, 6, 7]
    # b = [1, 1, 1, 1, 2, 2, 2, 2, 8, 12, 13]
    # print(intersection(a, b))
    # for i in range(0, 1000):
    #     test = random1dArray(100)
    #     selectionSort(test)
    #     if not isSortedAsc(test):
    #         print(test)
    # print(isSortedAsc(selectionSort(test)))
    #
    # test = [15, 12, 2, 4, 7, 9, 1, 20, 14, 52, 6, 3, 5, 2]
    # print(test)
    # inplacePartition(test, 10)
    # print(test)
    # for i in range(9, 0, -1):
    #     print(i)
    # insertionSort(test1)
    # print(test1)
    # for i in range(0, 10):
    #     test = random1dArray(100)
    #     insertionSort(test)
    #     if not isSortedAsc(test):
    #         print(test)
    # stoogeSort(test, 0, len(test)-1)
    # print(test)
    # given = polarRandom1dArray(10)
    # a = radixSort(given, 10)
    # print(a)
    # print(maxSubArray(a))

    global A
    A = Node(5, Color.BLACK)
    test = [3, 2, 8, 19, 23, 12]
    for i in test:
        RBInsert(A, i)
        print("Head is : ", A)
        traverseInOrder(A)
    print(B(A))
    # print()
    # traversePreOrder(A)
    # print()
    # traversePostOrder(A)
    # print(A, A.left, A.left.left, A.right, A.right.right, A.right.right.left, A.right.right.right)
    # print(A, A.color)
    # print(A.right, A.right.color)
    # print(A.left, A.left.color)
    # x = 3141592653589793238462643383279502884197169399375105820974944592
    # y = 2718281828459045235360287471352662497757247093699959574966967627
    # # print(karatsubaMultiplication(x, y, 64))
    # print(x*y == karatsubaMultiplication(x, y, 64))
    # B = Node(3, A, None, None)
    # C = Node(7, A, None, None)
    # D = Node(2, B, None, None)
    # E = Node(4, B, None, None)
    # F = Node(8, C, None, None)
    # G = Node(1, D, None, None)
    # H = Node(4.5, E, None, None)
    # A.set_children(B, C)
    # B.set_children(D, E)
    # C.set_children(F)
    # D.set_children(G)
    # E.set_children(None, H)

    # insert(A, 4.5)

    # deleteFromTree(A, 3)
    # deleteFromTree(A, 5)
    # traversePostOrder(A)
    # print("/")
    # traverseInOrder(A)
    # print("/n")
    # traversePreOrder(A)

    # for i in range(0, 1000):
    #     k = int(random.random()*len(test))
    #     if select(test, k) != mergeSort(test)[k-1]:
    #         print(select(test, k), mergeSort(test)[k-1])
    #         print(test)
    #         print(mergeSort(test))

    # for i in range(0, 10):
    #     rand = int(random.random()*len(test))
    #     print(mergeSort(test)[rand], select(test, rand))
    #     print(mergeSort(test))
