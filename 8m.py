
#8 a
from gensim.models import Word2Vec
sentences = [
    ["i", "love", "python"],
    ["python", "is", "easy"],
    ["i", "love", "machine", "learning"]
]
model = Word2Vec(sentences,
                 vector_size=5,
                 window=2,
                 min_count=1)
print("Embedding for 'python':")
print(model.wv['python'])
print("\nSimilar words to 'python':")
print(model.wv.most_similar('python'))

#8b 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
docs = [
    "machine learning is useful",
    "deep learning uses neural networks",
    "python programming is easy"
]
data = [TaggedDocument(d.split(), [i]) for i, d in enumerate(docs)]
model = Doc2Vec(data, vector_size=5, min_count=1, epochs=50)
v1 = model.dv[0]
v2 = model.dv[1]
sim = cosine_similarity([v1], [v2])
print("Similarity:", sim)
new_doc = "machine learning and neural networks"
new_vector = model.infer_vector(new_doc.split())
print("Most Similar Document:")
print(model.dv.most_similar([new_vector]))
