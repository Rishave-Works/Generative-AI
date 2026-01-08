from sklearn.feature_extraction.text import CountVectorizer
corpus = ["I love biryani", "I love jalebi"]
vector = CountVectorizer()
X = vector.fit_transform(corpus)
print(X) # it gives sparse matrix
print("Sparse matrix converted to array")
print(X.toarray())
print(vector.get_feature_names_out())