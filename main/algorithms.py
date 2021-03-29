import math

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


def strassen(b, c):
    print()


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
    result = np.empty((len(m1)+len(m2), len(m3) + len(m1)), type(m1[0][0]))
    result.fill(0)
    result[:int(len(m1)), :int(len(m1[0]))] = m1 + m4 - m5 + m7
    result[:int(len(m1) ), int(len(m1[0]) ):] = m3 + m5
    result[int(len(m1) ):, :int(len(m1[0]) )] = m2 + m4
    result[int(len(m1) ):, int(len(m1[0]) ):] = m1 + m3 - m2 + m6
    return result


def turnSquare(a, l):
    n = max(len(a), len(a[0]), l)
    n = int(math.pow(2, math.ceil(math.log(n, 2))))
    zero = np.zeros((n, n), type(a[0][0]))
    zero[:len(a), :len(a[0])] = a
    return zero


if __name__ == "__main__":
    a = [[4, 2, 6, 7], [6, 7, 8, 1], [4, 5, 4, 2]]
    b = [[4, 6], [6, 1]]
    a = turnSquare(a, len(b))
    b = turnSquare(b, len(a))
    if len(a) != len(b):
        exit(-1)
    print(strassen(a, b))
