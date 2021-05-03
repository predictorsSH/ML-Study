from common import data_load, preprocessing, _fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
#data load
train_df, test_df, train_x, train_target, test_x = data_load()

#data preprocessing
train_x_list, train_target, test_x_list = preprocessing(train_x, train_target, test_x)

def train_fasttext():
    full_data=list()
    for i in range(len(train_x_list)):
        for j in (train_x_list[i]):
            full_data.append(j)

    model = _fasttext(full_data)
    return model

model = train_fasttext()
embedding_dim = 100
zero_vector = np.zeros(embedding_dim)


class TextRank():

    def __init__(self, sentences,word_embedding_model,embedding_dim=100):

        self.sentences = sentences
        self.model = word_embedding_model
        self.embedding_dim = embedding_dim

    def calculate_sentence_vector(self,sentence):
        if len(sentence) != 0 :
            return sum([model.wv[word] for word in sentence])/len(sentence)
        else:
            return zero_vector

    def sentence_to_vectors(self):
        return [self.calculate_sentence_vector(sentence) for sentence in self.sentences]

    def similarity_matrix(self):

        sentence_vectors = self.sentence_to_vectors()

        sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)]) #유사도 행렬 크기
        for i in range(len(sentence_vectors)):
            for j in range(len(sentence_vectors)):
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, self.embedding_dim),
                                                  sentence_vectors[j].reshape(1,self.embedding_dim))[0,0]
        return sim_mat

    #그래프 그려보기
    def draw_graphs(self):
        sim_matrix = self.similarity_matrix()

        nx_graph= nx.from_numpy_array(sim_matrix)
        plt.figure(figsize=(10,10))
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, with_labels=True, font_weight='bold')
        nx.draw_networkx_edge_labels(nx_graph, pos, font_color='red')
        plt.show()

    def calculate_score(self):
        sim_matrix = self.similarity_matrix()

        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        sorted_scores = sorted(scores.items() , key=lambda item : item[1], reverse=True)
        print('scores :', sorted_scores)
        return scores, sorted_scores


#test
def test(tokenized_sentences, model, embedding_dim, original_sentence):
    tr = TextRank(tokenized_sentences,model,embedding_dim)
    scores, sorted_scores = tr.calculate_score()

    sentences=original_sentence

    print('original:')
    for i in range(len(sentences)):
        print(sentences[i])

    print('')
    print('summary:')
    for i in [i[0] for i in sorted_scores[:3]]:
        print(sentences[i])

test(train_x_list[0], model, 100, train_df['article_original'][0])

