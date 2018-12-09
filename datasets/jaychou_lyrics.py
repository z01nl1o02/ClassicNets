# -*- coding:utf-8 -*
from mxnet import nd
import random, zipfile,os

class JAYCHOU_LYRICS:
    def __init__(self,root='./', num_max=10000):
        with zipfile.ZipFile(os.path.join(root,'jaychou_lyrics.txt.zip')) as zin:
            with zin.open("jaychou_lyrics.txt") as f:
                corpus_chars = f.read().decode("utf-8")
        corpus_chars = corpus_chars.replace('\n',' ').replace('\r',' ')

        if num_max > 0 and len(corpus_chars) > num_max:
            corpus_chars = corpus_chars[0:num_max]


        idx_to_char = list(set(corpus_chars))
        char_to_idx = dict([(char,i) for i, char in enumerate(idx_to_char)])
        vocab_size = len(char_to_idx)
        corpus_indices = [char_to_idx[char] for char in corpus_chars]


        self.corpus_indices = corpus_indices
        self.corpus_chars = corpus_chars
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.vocab_size = vocab_size

    @property
    def corpus_indices(self):
        return self.corpus_indices
    @property
    def corpus_chars(self):
        return self.corpus_chars
    @property
    def idx_to_char(self):
        return self.idx_to_char
    @property
    def char_to_idx(self):
        return self.char_to_idx
    @property
    def vocab_size(self):
        return self.vocab_size

def data_iter_random(corpus_indices,batch_size,num_steps,ctx=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos+num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i:i+batch_size]
        X = [_data(j*num_steps) for j in batch_indices]
        Y = [_data(j*num_steps + 1) for j in batch_indices]
        yield nd.array(X,ctx), nd.array(Y,ctx)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0:batch_size*batch_len].reshape( (
        batch_size, batch_len
    ))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i:i+num_steps]
        Y = indices[:, i+1:i+num_steps+1]
        yield X,Y


def to_onehot(X,size): #one column, one sample
    return [nd.one_hot(x, size) for x in X.T]



if 0:
    ds = JAYCHOU_LYRICS()
    #print ds.corpus_chars
    print ds.char_to_idx['想'.decode('utf-8')]
    print(r' -' + r''.join(ds.idx_to_char[0:10]))
    #my_seq = list(range(30))
    #for X,Y in data_iter_random(my_seq, batch_size=1, num_steps=6):
    #   print("X: ",X)
    #   print("Y: ",Y)
