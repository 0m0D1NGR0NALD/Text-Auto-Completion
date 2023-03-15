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
# Display first 300 letters
print("First 300 letters of the data")
print("-------")
display(data[0:300])
print("-------")
# Display last 300 letters
print("Last 300 letters of the data")
print("-------")
display(data[-300:])
print("-------")

# Function to split data by linebreak
def split_to_sentences(data):
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences]
    return sentences
