from math import log, ceil

if __name__ == '__main__':
    # n, k = map(int, input().split())
    # h = 1
    # c = 1
    # while (h + k) % n != 1:
    #     c += 1
    #     h += k
    # print(c)
    # lengths = []
    # widths = []
    # for i in range(0, 3):
    #     a, b = map(float, input().split())
    #     lengths.append(a)
    #     widths.append(b)
    # if lengths[0] != lengths[1]:
    #     lengths[1], widths[1], lengths[2], widths[2] = lengths[2], widths[2], lengths[1], widths[1]
    # d = abs(widths[0] - widths[1])
    # print(int(lengths[2]), int(widths[2] - d) if widths[2] == max(widths[1], widths[0]) else int(widths[2] + d))
    # n = int(input())
    # k = int(input())
    # c = 0
    # c += (n + 1)//k

    # n = int(input())
    # old = 0
    # c = 0
    # for i in range(0, n):
    #     new = int(input())
    #     if i > 0:
    #         if old != new:
    #             c += 1
    #     old = new
    # print(c)

    # n = int(input())
    # first = input()
    # sec = input()
    # c = 0
    # for i in range(0, n):
    #     if first[i] != sec[i]:
    #         c += 1
    # print(c)

    n = int(input())
    res = 1
    for i in range(2, n+1):
        res *= i
    k = int(log(res, 10))+1
    
    # n = int(input())
    # a = []
    # total = 0
    # for i in range(0, n):
    #     a.append(int(input()))
    #     total += a[i]
    # res = 0
    # for x in a:
    #     res += abs((total//n)-x)
    # print(res//2)
