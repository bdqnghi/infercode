import nltk
words = set(nltk.corpus.words.words())

with open("temp_tokens.csv", "r") as f1:
    lines = f1.readlines()
    for sent in lines:
        sent = sent.replace("\n", "")
        sent = " ".join(w for w in nltk.wordpunct_tokenize(sent) if w.lower() in words or not w.isalpha())

        with open("temp_tokens_2.csv", "w") as f2:
            f2.write(sent)
            f2.write("\n")