from SJAYA import SJAYA
import numpy as np
import boto3
import os

# Get the current working directory
base_path = os.getcwd() + "/"

# Bucket name
bucket_name = input("Enter the bucket name: ")

paraRange = [[0, 0], [1, 1]]
numOfPara = len(paraRange[0])
numOfDesigns = 10
popSize = 10

s = SJAYA(numOfDesigns, popSize)
for i in range(5):
    s.PerformSJAYA(numOfPara, paraRange, save_path=f"output_plot_{i}.png")
    print(f"Sampling {i} done!")

# Initialize the S3 client outside the loop
s3 = boto3.client('s3')

# Loop over the range of file indices
for i in range(5):  # This will iterate over numbers 0 through 4
    file_name = f"output_plot_{i}.png"
    file_path = base_path + file_name

    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, file_name)
    print(f"File {file_name} added to bucket!")

