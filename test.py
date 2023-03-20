from data import train_data,test_data,preprocess_data
from IPython.display import display

# Function to predict multiple suggestions by looping over various n-gram models
def get_suggestions(previous_n_gram,n_gram_counts_list,vocabulary,k=1.0,start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]

        suggestion = suggest_a_word(previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary,k=k,start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data,minimum_freq)

n_gram_counts_list = []
for n in range(1,6):
    print("Computing n-gram counts with n=",n, "...")
    n_model_counts = count_n_grams(train_data_processed,n)
    n_gram_counts_list.append(n_model_counts)
