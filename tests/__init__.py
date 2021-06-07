
if __name__ == '__main__':
    inp = []
    for i in range(0, 2):
        inp.append(int(input))
    if inp[0]**2+inp[1]**2==inp[2]**2 or inp[1]**2+inp[2]**2==inp[0]**2 or inp[0]**2+inp[2]**2==inp[1]**2:
        print("YES")
    else:
        print("NO")