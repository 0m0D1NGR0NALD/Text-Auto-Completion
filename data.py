import math
import random
import numpy as np
import pandas as pd
from IPython.display import display
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.data.path.append('.')

# Opening data file to read data in file
with open("en_US.twitter.txt", "r") as file:
    data = file.read()

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
    # Initialize empty list
    split_sentences = []
    # Loop through sentences
    for sentence in sentences:
        # Convert items to lowercase for uniformity
        sentence = sentence.lower()
        # Separate sentence into a list of tokens
        split_sentence = word_tokenize(sentence)
        # Add tokens to list
        split_sentences.append(split_sentence)
     # Return list of tokenized sentences
    return split_sentences

# Function to get split sentences
def get_split_sentences(data):
    # Split data
    sentences = split_to_sentences(data)
    # Tokenize split data
    split_sentence = split_sentences(sentences)
    # Return tokenized data
    return split_sentence

# Obtain tokenized data
tokenized_data = get_split_sentences(data)
# Set random seed
random.seed(24)
# Shuffle data
random.shuffle(tokenized_data)
# Set split ratio
split_size = int(len(tokenized_data)*0.8)
# Define train set
train_data = tokenized_data[0:split_size]
# Define test set
test_data = tokenized_data[split_size:]

# Function to count words to discover frequently used words
def count_words(tokenized_data):
    word_counter = {}
    for sentence in range(len(tokenized_data)):
        # Loop through tokens in the sentence
        for token in (tokenized_data[sentence]):
            if token not in word_counter.keys():
                word_counter[token] = 1
            else:
                word_counter[token] += 1
    return word_counter

# Function to select frequently used words
def get_words_by_threshold_frequency(tokenized_data,threshold):
    common_vocabulary = []
    word_count = count_words(tokenized_data)
    for word,count in word_count.items():
        if count >= threshold:
            common_vocabulary.append(word)
    return common_vocabulary

# Function to replace unknown words
def replace_oov_by_unk_words(tokenized_data,vocabulary,unknown_token="<unk>"):
    vocabulary = set(vocabulary)
    replaced_tokenized_data = []

    # Loop through tokenized sentences
    for sentence in tokenized_data:
        replaced_data = []
        # Loop through tokens
        for token in sentence:
            # Check if token in commonly used words
            if token in vocabulary:
                replaced_data.append(token)
            else:
                replaced_data.append(unknown_token)
        replaced_tokenized_data.append(replaced_data)
    return replaced_tokenized_data
