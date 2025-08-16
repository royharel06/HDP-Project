import gensim.downloader as api
import numpy as np

# Options: 'glove-wiki-gigaword-300', 'word2vec-google-news-300', 'fasttext-wiki-news-subwords-300'
kv = api.load('glove-wiki-gigaword-300')   # ~400k words × 300D

# 1. Basic info
print(f"Vocabulary size: {len(kv.index_to_key):,}")
print(f"Vector dimension: {kv.vector_size}")

# 2. See the first 10 words in the vocab
print("First 10 words:", kv.index_to_key[:10])

# 3. Get the vector for a specific word
word = "king"
vector = kv[word]
print(f"\nVector for '{word}':")
print(vector[:10], "...")  # print only first 10 dimensions

# 4. Find most similar words
print("\nMost similar to 'king':")
for similar_word, score in kv.most_similar("king", topn=5):
    print(f"{similar_word:10}  cosine sim = {score:.3f}")

# 5. Analogy example: king - man + woman ≈ queen
result = kv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)
print("\n'king' - 'man' + 'woman' ≈ ?")
for word, score in result:
    print(f"{word:10}  cosine sim = {score:.3f}")