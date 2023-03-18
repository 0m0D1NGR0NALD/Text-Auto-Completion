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
