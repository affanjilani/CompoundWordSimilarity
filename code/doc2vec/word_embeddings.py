# Dataset1 from https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.Xds33ehKg2x

import gzip
import gensim
import os
from tqdm import tqdm

# Generates a word2vec model
def generate_word_2_vec():

    data = [os.path.join('../../raw_data/reviews_data.txt.gz')]

    documents = []

    for f in data:
        documents += list(generate_documents(f))
    print('doc length', len(documents))

    model = gensim.models.Word2Vec(documents,size=150,window=10,min_count=2,workers=10)

    print('*************************here*************************')
    
    model.train(documents, total_examples=len(documents), epochs=10)

    model.save(os.path.join('./opin/opin_rank_vectors'))

    return model

# Generates the list of documents that get used to train the word2vec model
def generate_documents(data_file):
    with gzip.open(data_file,'rb') as f:
        for i, line in tqdm(enumerate(f)):
            yield gensim.utils.simple_preprocess(line)

if __name__ == '__main__':
    #model = generate_word_2_vec()
    model = gensim.models.Word2Vec.load('./opin/opin_rank_vectors')
    vecs = model.wv
    del model
    w1 = "asdf"
    print("Most similar to {0}".format(w1), vecs.most_similar(positive=w1))



