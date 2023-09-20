from SJAYA import SJAYA
import numpy as np

paraRange = [[0, 0],[1 ,1]]
numOfPara = len(paraRange[0])
numOfDesigns = 10
popSize = 10

s = SJAYA(numOfDesigns, popSize)
s.PerformSJAYA(numOfPara,paraRange)
ptint("Its done!")
"""
# create the 2D array
arr = np.zeros((rows, columns))
for i in range(columns):
    arr[:, i] = np.random.uniform(paraRange[0,i], paraRange[1,i], size=rows)

print(arr)
"""
