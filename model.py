import pandas as pd
import numpy as np

# Building an N-gram based language models

# Function that computes n-grams count for each arbitrary n number
def count_n_grams(data,n,start_token="<s>",end_token="<e>"):
    # Initialize dictionary for n_grams and their cummulative count
    n_grams = {}
    # Loop through data
    for sentence in range(len(data)):
        sentences = [start_token]*n + list(data[sentence]) + [end_token]
        sentences = tuple(sentences)
        for i in range(0,len(sentences)-n+1):
            # Obtain n-gram from i to i+n
            n_gram = sentences[i:i+n]
            # Check if n-gram is in dictionary
            if n_gram in n_grams.keys():
                # Increment count
                n_grams[n_gram] += 1
            else:
                # if not, set count to one
                n_grams[n_gram] = 1
        # Return dictionary of n_grams and their cummulative count
        return n_grams
        
# Function that calculates the numerator and denominator to obtain estimated probability of interest       
def estimate_probability(word,previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k=1.0):
    # Check if previous n-gram is of datatype "list"
    if type(previous_n_gram) == list:
        # Convert list to tuple
        previous_n_grams = tuple(previous_n_gram)
    else:
        # Otherwise, first convert to list then to tuple
        previous_n_grams = tuple([previous_n_gram])
    previous_n_gram_count = n_gram_counts.get(previous_n_grams,0)
    # Apply k-smoothing to handle zero counts in denominator
    denominator = previous_n_gram_count + vocabulary_size*1
    n_plus1_gram = previous_n_grams + tuple([word])
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram,0)
    # Apply smoothing to numerator
    numerator = n_plus1_gram_count + k
    # Calculating probability
    probability = numerator/denominator
    # Return calculated probability
    return probability

def estimate_probabilities(previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary,k=1.0):
    previous_n_gram = previous_n_gram
    # Add <e> <unk> to the vocabulary <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + ["<e>","<unk>"]
    vocabulary_size = len(vocabulary)
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word,previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k=k)
        probabilities[word] = probability
    return probabilities

def make_count_matrix(n_plus1_gram_counts, vocabulary):
    vocabulary = vocabulary + ["<e>","<unk>"]
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    row_index = {n_gram:i for i,n_gram in enumerate(n_grams)}
    col_index = {word:j for j,word in enumerate(vocabulary)}
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow,ncol))
    for n_plus1_gram,count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i,j] = count
    count_matrix = pd.DataFrame(count_matrix,index=n_grams,columns=vocabulary)
    return count_matrix

def make_probability_matrix(n_plus1_gram_counts,vocabulary,k):
    count_matrix = make_count_matrix(n_plus1_gram_counts,vocabulary)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1),axis=0)
    return prob_matrix

def calculate_perplexity(sentences,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k=1.0):
    n = len(list(n_gram_counts.keys())[0])
    sentence = ["<s>"]*n + sentences + ["<e>"]
    sentence = tuple(sentence)

    N = len(sentence)
    product_pi = 1.0
    for i in range(n,N):
        n_gram = sentence[i-n]
        word = sentence[i]
        probability = estimate_probability(word,n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k)
        product_pi *= 1/probability

    # Nth root of the product
    perplexity = product_pi**(1/float(N))
    perplexity = float(perplexity)

    return perplexity

def suggest_a_word(previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary,k=1.0,start_with=None):
    probabilities = estimate_probabilities(previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary,k=k)
    suggestion = None
    max_prob = 0
    for word,prob in probabilities.items():
        if start_with != None:
            if word.startswith(start_with)==False:
                continue
        if prob > max_prob:
            suggestion = word
            max_prob = prob
    return suggestion,max_prob
