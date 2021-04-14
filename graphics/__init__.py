import matplotlib.pyplot as plt
import random
import time
import timeit
from functools import partial


def isSortedAsc(a):
    for i in range(0, len(a)-1):
        if a[i] > a[i + 1]:
            return False
    return True


def random1dArray(n=5):
    test = []
    for j in range(0, n):
        test.append(int(random.random() * 100) + 1)
    return test


def chooseRandomPivot(n):
    return int(random.random()*n)


def quickSort(a):
    start = time.time_ns()
    if len(a) < 3:
        return mergeSort(a)
    L, mid, R = partition(a, chooseRandomPivot(len(a)))
    end = time.time_ns()
    return quickSort(L) + [mid] + quickSort(R)


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


def BogoSort(a):
    while not isSortedAsc(a):
        random.shuffle(a)


def runningTimeSort(fn, maxIn, nTests):
    times = []
    given = random1dArray(maxIn)
    inputs = range(0, maxIn+1)
    for i in inputs:
        nthTimer = timeit.Timer(partial(fn, given[:i]))
        times.append(nthTimer.timeit(nTests))
    return times


if __name__ == "__main__":
    limit = 1000
    inputs = range(0, limit + 1)

    # mTimes = runningTimeSort(mergeSort, limit, 10)
    bTimes = runningTimeSort(BogoSort, 30, 100)
    # qTimes = runningTimeSort(quickSort, limit, 10)
    # limit = 10
    # inputs = range(1, limit+1)
    # given = random1dArray(limit)
    #
    # for i in inputs:
    #     b = given[:i]
    #     bTimes.append(BogoSort(b))
    #     b = given[:i]
    #     # qTimes.append(timeit.timeit(quickSort(b)))

    # plt.plot(inputs, mTimes, "--", label="Merge Sort", color='Green')
    plt.plot(range(0, 11), bTimes, "-", label="Bogo Sort", color='Red')
    # plt.legend(inputs, qTimes, "--", label="Quick Sort", color='Blue')

    plt.xlabel('Input')
    plt.ylabel('Time(s)')
    plt.title('Running time of BogoSort for different inputs')
    plt.show()
