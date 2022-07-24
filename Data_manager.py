import numpy as np
import random
import re
import linecache
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical


class Vocab:
    def __init__(self, word2id, unk_token):  # o.15 coming from utility.GeneratorPretrainingGenerator.init
        self.word2id = dict(word2id)  # o.16 init word2id with defult_dict parameters=>{'<PAD>': 0, '<S>': 1, '</S>': 2, '<UNK>': 3}
        self.id2word = {v: k for k, v in self.word2id.items()}  # o.17 id2word is the opposite of word2id=>{'0: <PAD>', .....}
        self.unk_token = unk_token

    def build_vocab(self, sentences, min_count=1):  # o.19 coming from utility.GeneratorPretrainingGenerator.init
        word_counter = {}  # o.20 here we will store all the words and how much are there of each word
        for sentence in sentences:
            for word in sentence:  # making array of words and how much of them are there
                word_counter[word] = word_counter.get(word, 0) + 1  # o.21 counting +1 each word

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):  # o.22 and 6 below are just givid ids to words and the oposite
            if count < min_count:  # in word2id {....., '<UNK>': 3, 'the': 4 } and so one
                break  # in id2word {.....,  3:'<UNK>', 4: 'the' } and so one
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

        self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}  # o.23 create list of words

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.word2id[self.unk_token] for word in sentence]


def load_data(file_path):  # o.19 coming from utility.GeneratorPretrainingGenerator.init
    data = []
    for line in open(file_path, encoding='utf-8'):
        line = line.lower()
        line = re.sub('\W+', ' ', line)
        words = line.strip().split()
        if (len(words) > 0):
            data.append(words)

    return data


def sentence_to_ids(vocab, sentence, UNK=3):
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    return ids


def pad_seq(seq, max_length, PAD=0):
    seq += [PAD for i in range(max_length - len(seq))]
    return seq


class GeneratorDataManeger(Sequence):  # call from TRAIN o.11
    def __init__(self, parameters):
        self.parameters = parameters
        self.len = self.__len__()  # o.25 claculate lan of a batch?
        self.reset()  # o.26 mess lines order in text and idx=0

    def __len__(self):  ##o.25 you see above
        return self.parameters.n_data // self.parameters.batch_size

    def __getitem__(self, idx):

        x, y_true = [], []
        start = idx * self.parameters.batch_size + 1

        end = (idx + 1) * self.parameters.batch_size + 1

        max_length = 0
        for i in range(start, end):
            if self.parameters.shuffle:
                idx = self.shuffled_indices[i]
            else:
                idx = i
            sentence = linecache.getline(self.parameters.path_pos, idx)  # str

            sentence = sentence.lower()
            words = sentence.strip().split()  # list of str
            ids = sentence_to_ids(self.parameters.vocab, words)  # list of ids

            ids_x, ids_y_true = [], []

            ids_x.append(self.parameters.BOS)
            ids_x.extend(ids)
            ids_x.append(self.parameters.EOS)  # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)

            ids_y_true.extend(ids)
            ids_y_true.append(self.parameters.EOS)  # ex. [8, 10, 6, 3, EOS]
            y_true.append(ids_y_true)

            max_length = max(max_length, len(ids_x))

        if self.parameters.seq_len is not None:
            max_length = self.parameters.seq_len

        for i, ids in enumerate(x):
            x[i] = x[i][:max_length]
        for i, ids in enumerate(y_true):
            y_true[i] = y_true[i][:max_length]

        x = [pad_seq(sen, max_length) for sen in x]
        x = np.array(x, dtype=np.int32)

        y_true = [pad_seq(sen, max_length) for sen in y_true]
        y_true = np.array(y_true, dtype=np.int32)
        y_true = to_categorical(y_true, num_classes=self.parameters.word_count)

        return (x, y_true)

    def __iter__(self):
        return self

    def next(self):
        if self.parameters.idxG >= self.len:
            self.reset()
            raise StopIteration
        x, y_true = self.__getitem__(self.parameters.idxG)
        self.parameters.idxG += 1
        return (x, y_true)

    def reset(self):  # o.26 call from utility.GeneratorPretrainingGenerator.init
        self.parameters.idxG = 0
        if self.parameters.shuffle:  # shuffle is boolean
            self.shuffled_indices = np.arange(self.parameters.n_data)  # list of numbers os lines [1,2,4....,n_data]
            random.shuffle(self.shuffled_indices)  # then take shuffled_indices and randomly mess the order of lines

    def on_epoch_end(self):
        self.reset()
        pass


class DiscriminatorDataManager(Sequence):
    def __init__(self, parameters):  # o.27 coming from train.Trainer._init
        self.parameters = parameters
        counter = 0
        with open(parameters.path_neg, 'r', encoding='utf-8') as f:
            counter = sum(1 for line in f)
        self.parameters.n_dataD = self.parameters.n_data + counter
        self.len = self.__len__()
        self.reset()

    def __len__(self):  # same
        return self.parameters.n_dataD // self.parameters.batch_size

    def __getitem__(self, idx):
        X, Y = [], []
        start = idx * self.parameters.batch_size + 1
        end = (idx + 1) * self.parameters.batch_size + 1
        max_length = 0
        for i in range(start, end):
            idx = self.indicies[i]
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                sentence = linecache.getline(self.parameters.path_pos, idx)  # str
            elif is_pos == 0:
                sentence = linecache.getline(self.parameters.path_neg, idx)  # str
            words = sentence.strip().split()  # list of str
            ids = sentence_to_ids(self.parameters.vocab, words)  # list of ids

            x = []
            x.extend(ids)
            x.append(self.parameters.EOS)  # ex. [8, 10, 6, 3, EOS]
            X.append(x)
            Y.append(is_pos)

            max_length = max(max_length, len(x))

        if self.parameters.seq_len is not None:
            max_length = self.parameters.seq_len

        for i, ids in enumerate(X):
            X[i] = X[i][:max_length]

        X = [pad_seq(sen, max_length) for sen in X]
        X = np.array(X, dtype=np.int32)

        return (X, Y)

    def __iter__(self):
        return self

    def next(self):
        if self.parameters.idxD >= self.len:
            self.reset()
            raise StopIteration
        X, Y = self.__getitem__(self.parameters.idxD)
        self.parameters.idxD += 1
        return (X, Y)

    def reset(self):
        self.parameters.idxD = 0
        pos_indices = np.arange(start=1, stop=self.parameters.n_data + 1)
        neg_indices = -1 * np.arange(start=1, stop=self.parameters.n_dataD + 1)
        self.indicies = np.concatenate([pos_indices, neg_indices])
        if self.parameters.shuffle:
            random.shuffle(self.indicies)

    def on_epoch_end(self):
        self.reset()
        pass
