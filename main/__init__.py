import random


def fillBinary():
    with open("D:\\Users\\shahram\\test.txt", 'w') as file:
        for i in range(0, 1001):
            c = ""
            for j in range(0, 8):
                c = c + str(int(2 * random.random()))
            file.write(c+" \n")


if __name__ == "__main__":
    fillBinary()
