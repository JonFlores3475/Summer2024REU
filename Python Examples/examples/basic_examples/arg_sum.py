import sys

list = sys.argv

if len(list) != 3:
    sys.exit
else:
    print((int(list[1]) + int(list[2])))