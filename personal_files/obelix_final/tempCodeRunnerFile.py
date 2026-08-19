# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
# print("Try programiz.pro")

import numpy as np

class PER:
    def __init__(self):
        self.buf = []
        
    def add(self, a, b):
        item = (a,b)
        self.buf.append(item)

replay = PER()
a = np.zeros(3)
b = np.zeros(4)
replay.add(a,b)
a = b
for r in replay.buf:
    print(r)
