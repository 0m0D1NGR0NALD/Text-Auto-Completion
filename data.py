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
    # Split data using linebreak to create list of sentences
    sentences = data.split("\n")
    # Remove any white spaces in split data >> sentences
    sentences = [s.strip() for s in sentences]
    # Elimnate empty items in list of sentences
    sentences = [s for s in sentences if len(s) > 0]
    # Return list of sentences
    return sentences

# Function to split sentence into list of words
def split_sentences(sentences):
    split_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        split_sentence = nltk.word_tokenize(sentence)
        split_sentences.append(split_sentence)
     return split_sentences

# Function to get split sentences
def get_split_sentences(data):
    sentences = split_to_sentences(data)
    split_sentences = split_sentences(sentences)
    return split_sentences
