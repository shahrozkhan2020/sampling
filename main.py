from SJAYA import SJAYA
import numpy as np
import boto3

paraRange = [[0, 0],[1 ,1]]
numOfPara = len(paraRange[0])
numOfDesigns = 10
popSize = 10

s = SJAYA(numOfDesigns, popSize)
for i in range(5):
    s.PerformSJAYA(numOfPara, paraRange, save_path=f"output_plot_{i}.png")
    print(f"output_plot_{i}.png")

base_path = "/home/ubuntu/test/sampling/"

# Initialize the S3 client outside the loop, so you don't have to do it multiple times
s3 = boto3.client('s3')

# Loop over the range of file indices
for i in range(5):  # This will iterate over numbers 1 through 5
    file_name = f"output_plot_{i}.png"
    file_path = base_path + file_name

    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, "samplingtestbucket", file_name)
    print(f"File {file_name} added to bucket!")

"""
# create the 2D array
arr = np.zeros((rows, columns))
for i in range(columns):
    arr[:, i] = np.random.uniform(paraRange[0,i], paraRange[1,i], size=rows)

print(arr)
"""
