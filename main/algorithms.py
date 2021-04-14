import math
import random
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


def maxSubArrayNaive(a):
    maximum = a[0]
    s = 0
    e = 0
    for i in range(0, len(a)):
        sum = 0
        for j in range(i, len(a)):
            sum += a[j]
            if sum > maximum:
                maximum = sum
                s = i
                e = j
    return Result(maximum, s, e)


def maxSubArray(a):
    pass


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


class Node:
    def __init__(self, key, parent, left, right):
        self.left = left
        self.right = right
        self.key = key
        self.parent = parent

    def set_parent(self, parent):
        self.parent = parent

    def set_key(self, value):
        self.key = value
    
    def set_right(self, right):
        self.right = right

    def set_left(self, left):
        self.left = left

    def print_instance_name(self):
        print(self.__class__.__name__)


def search(current, val):
    if current is None:
        return None
    if current.value == val:
        return current
    le = search(current.left, val)
    if le is not None:
        return le
    else:
        return search(current.right, val)


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
    return int(random.random()*n)


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
        return mergeSort(a)[k-1]
    p = chooseRandomPivot(len(a))
    L, mid, R = partition(a, p)
    if len(L) == k:
        return mid
    elif len(L) > k:
        return select(L, k)
    else:
        return select(R, k - len(L) -1)


def isSortedAsc(a):
    for i in range(0, len(a)-1):
        if a[i] > a[i + 1]:
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


def random1dArray(n=5):
    test = []
    for j in range(0, n):
        test.append(int(random.random() * 100) + 1)
    return test


def countingSort(a, n=10):
    buckets = [[] for i in range(0, 10 ** n)]
    for i in range(0, len(a)):
        buckets[a[i]].append(a[i])
    result = []
    print(buckets)
    for i in range(0, len(a)):
        result += buckets[i]
    return result


def radixSort(a, n=10):
    result = [[] for i in range(0, 10)]
    for i in range(0, len(a)):
        result[a[i]%10].append(a[i])
    print(result)


if __name__ == "__main__":
    test = random1dArray(10)
    print(test)
    print(countingSort(test, 4))
    print(len(test), len(countingSort(test, 4)))
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
    # test = [20, 14, 52, 16, 37, 8, 23, 13, 64, 43, 76, 32, 21, 12, 4]
    # stoogeSort(test, 0, len(test)-1)
    # print(test)
    # A = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    # print(maxSubArray(A))
    # d = [5, 2, 3, 4, 6, 7, 8]
    # minMult(d)
    # er = Node(53, None, None)
    # el = Node(143, None, None)
    # dr = Node(34, None, None)
    # dl = Node(143, None, None)
    # cr = Node(14, None, None)
    # cl = Node(30, None, None)
    # br = Node(7, el, er)
    # bl = Node(42, dl, dr)
    # ar = Node(43, cl, cr)
    # al = Node(15, bl, br)
    # head = Node(22, al, ar)
    # print(search(head, 7))


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
