import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm
import json
from gensim.models import FastText
import pickle

def data_load():
    with open('train.jsonl', 'r') as f:
        json_list = list(f)

    trains = []
    for json_str in json_list:
        line = json.loads(json_str)
        trains.append(line)

    train_df = pd.DataFrame(trains)

    with open('extractive_test_v2.jsonl', 'r') as f:
        json_list=list(f)

    tests = []
    for json_str in json_list:
        line = json.loads(json_str)
        tests.append(line)

    test_df = pd.DataFrame(tests)

    return train_df, test_df, train_df['article_original'], train_df['extractive'], test_df['article_original']


def preprocessing(train,y,test):
    twitter = Okt()
    train_x_list = list()
    for i in tqdm(range(len(train[:10000]))):
        li = list()
        for j in (train[i]):
            li.append(twitter.nouns(j))
        train_x_list.append(li)

    train_file_path = './train100.txt'

    with open(train_file_path, 'wb') as f:
        pickle.dump(train_x_list, f)

    test_x_list = list()
    for i in tqdm(range(len(test[:10000]))):
        li = list()
        for j in (test[i]):
            li.append(twitter.nouns(j))
        test_x_list.append(li)

    test_file_path = './test100.txt'

    with open(test_file_path, 'wb') as f :
        pickle.dump(test_x_list, f)


    ## load saved file
    # with open(train_file_path, 'rb') as lf:
    #     train_x_list = pickle.load(lf)
    #
    # with open(test_file_path, 'rb') as lf:
    #     test_x_list = pickle.load(lf)


    return train_x_list, y, test_x_list


def _fasttext(data):
    model = FastText(data,vector_size=100,window=5,workers=3,sg=1,min_count=2)
    return model

# def pretrained_model():
#     print('load pretrained model!')
#     ko_model = models.fasttext.load_facebook_model('cc.ko.300.bin.gz')
#
#     print('done')
#     return ko_model


