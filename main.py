from SJAYA import SJAYA
import numpy as np
import boto3

paraRange = [[0, 0],[1 ,1]]
numOfPara = len(paraRange[0])
numOfDesigns = 10
popSize = 10

s = SJAYA(numOfDesigns, popSize)
s.PerformSJAYA(numOfPara, paraRange, save_path="output_plot.png")
print("sampling done!")

s3 = boto3.client('s3')
file_path = "/home/ubuntu/test/sampling/output_plot.png"
with open(file_path, "rb") as f:
    s3.upload_fileobj(f, "samplingtestbucket", "output_plot.png")
print("File added to bucket!")

"""
# create the 2D array
arr = np.zeros((rows, columns))
for i in range(columns):
    arr[:, i] = np.random.uniform(paraRange[0,i], paraRange[1,i], size=rows)

print(arr)
"""
