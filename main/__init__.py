import random
import re
import matplotlib.pyplot as mp
import numpy as np


def fillBinary():
    with open("D:\\Users\\shahram\\test.txt", 'w') as file:
        for i in range(0, 1001):
            c = ""
            for j in range(0, 8):
                c = c + str(int(2 * random.random()))
            file.write(c+" \n")
        file.write("\n  ")


def readIntFile():
    with open("E:\\6th_term\\FPGA-Lab\\4th_session\\samples.txt", 'r') as file:
        result = re.findall("\\d+", file.readline())
    return result


if __name__ == "__main__":
    result = readIntFile()
    with open("E:\\6th_term\\FPGA-Lab\\4th_session\\test.txt", 'w+') as file:
        for i in range(0, 64):
            file.write((bin(int(result[i]))+"")[2:].rjust(8, '0')+" \n")
