# doc2vec

doc2vec is a shallow, two-layer neural network that accepts a text corpus as an input, and it returns a set of vectors (aka embeddings); each vector is a numeric representation
of a given sentence, paragraph, or document.

One difference between w2v and d2v is that d2v requires you to create tagged documents. The tagged document expects you create a list of words and a tag for each document.
And the d2v model trains on top of those tagged documents. This tag is useful if you have this distinct group of documents. 
