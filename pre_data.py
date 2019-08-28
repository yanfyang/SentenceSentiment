import pandas as pd
import re
import jieba
import numpy as np

def prepare_data(negfile,posfile):
    neg = pd.read_excel(negfile, header=None)
    pos = pd.read_excel(posfile, header=None)
    neg['label'] = 0
    pos['label'] = 1
    data = neg.append(pos)
    shuffledata = data.sample(frac=1).reset_index(drop=True)
    shuffledata.rename(columns={0:'comm'},inplace=True)
    return shuffledata

def comm_clean(comm):   
    comm = re.sub(r'[ #]','',comm)
    cut = ' '.join(jieba.cut(comm,HMM=False))
    return cut

def build_vocab(data):
    vocab = set()
    for sen in data:
        vocab = vocab.union(set([word for word in sen.split()])) 
    vocab = list(vocab)
    vocab.extend(['PAD','UN'])
    word2id = {}
    id2word = {}
    for i,w in enumerate(vocab):
        word2id[w] = i
        id2word[i] = w    
    return word2id, id2word

def get_x(sen,word2id,max_len):
    '''
    Function: change a sentence to max_len limited word id sentence array
    Args:
        sen: string, seperated phrase sentence using jieba cut
        word2id: dict, word2id dict
        max_len: int, max_len of phrases in sentence        
    '''
    sen_id = [word2id[word] for word in sen.split()]
    if len(sen_id)<max_len:
        sen_id.extend([word2id['PAD']]*(max_len-len(sen_id)))
    return np.asarray(sen_id[:max_len])
    
def get_batch(X,Y,output_size,batch_size,word2id,max_len): 
    '''
    Function: Data Generateor for batch_x and batch_y
    Args:
        X: list, all sentences list,such as ['超级 好用 ， 以后 还 会 再来 的 。 ','这个 电影 一点儿 也 不 值得 看']
        Y：list, label list, such as [1,0]
        output_size: int, category size, 2 for this project
        batch_size: int
        word2id: dict
        max_len: int       
    '''
    for i in range(0,len(X),batch_size):
        batch_X = X[i:i+batch_size]
        batch_Y = Y[i:i+batch_size]
        embed_batch_X = []
        embed_batch_Y =[]
        for sen in batch_X:
            x = get_x(sen,word2id,max_len)
            embed_batch_X.append(x)
            
        for label in batch_Y:
            embed_batch_Y.append(np.eye(output_size)[label])
        yield embed_batch_X, embed_batch_Y

