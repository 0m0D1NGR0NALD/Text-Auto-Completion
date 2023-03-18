import pandas as pd
import numpy as np

def count_n_grams(data,n,start_token="<s>",end_token="<e>"):
    n_grams = {}
    for sentence in range(len(data)):
        print(sentence)
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
                # if not set count to one
                n_grams[n_gram] = 1
        return n_grams
        
def estimate_probability(word,previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k=1.0):
    if type(previous_n_gram) == list:
        previous_n_grams = tuple(previous_n_gram)
    else:
        previous_n_grams = tuple([previous_n_gram])
    previous_n_gram_count = n_gram_counts.get(previous_n_grams,0)
    # Apply k-smoothing to handle zero counts
    denominator = previous_n_gram_count + vocabulary_size*1
    n_plus1_gram = previous_n_grams + tuple([word])
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram,0)
    # Apply smoothing
    numerator = n_plus1_gram_count + k
    probability = numerator/denominator
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
