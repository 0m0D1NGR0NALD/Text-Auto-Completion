import math
import random
import numpy as np
import pandas as pd
from IPython.display import display
import nltk
nltk.data.path.append('.')

# Opening data file to read data in file
with open("en_US.twitter.txt", "r") as file:
    data = file.read()

# Print data type
print("Data type:", type(data))
# Print number of letters in dataset
print("Number of letters:", len(data))
# Display first 240 letters
print("First 240 letters of the data")
print("-------")
display(data[0:240])
print("-------")
# Display last 240 letters
print("Last 240 letters of the data")
print("-------")
display(data[-240:])
print("-------")

# Function to split data by linebreak
def split_to_sentences(data):
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences
