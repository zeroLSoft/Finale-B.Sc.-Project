import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
import tensorflow as tf
import pickle


class GeneratorLSTM():
    def __init__(self, parameters):   #o.32 coming from rl._init
        self.parameters=parameters
        self._build_gragh()   #o.33 
        self.reset_rnn_state() #o.34

    def _build_gragh(self):
        state_in = tf.placeholder(tf.float32, shape=(None, 1))
        h_in = tf.placeholder(tf.float32, shape=(None, self.parameters.generator_H))
        c_in = tf.placeholder(tf.float32, shape=(None, self.parameters.generator_H))
        action = tf.placeholder(tf.float32, shape=(None, self.parameters.word_count)) # onehot (B, V)
        reward = tf.placeholder(tf.float32, shape=(None, ))

        self.layers = []

        embedding = Embedding(self.parameters.word_count, self.parameters.generator_E, mask_zero=True, name='Embedding')
        out = embedding(state_in) # (B, 1, E)
        self.layers.append(embedding)

        lstm = LSTM(self.parameters.generator_H, return_state=True, name='LSTM')
        out, next_h, next_c = lstm(out, initial_state=[h_in, c_in])  # (B, H)
        self.layers.append(lstm)

        dense = Dense(self.parameters.word_count, activation='softmax', name='DenseSoftmax')
        prob = dense(out)    # (B, V)
        self.layers.append(dense)

        log_prob = tf.log(tf.reduce_mean(prob * action, axis=-1)) # (B, )
        loss = - log_prob * reward
        optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters.generator_lr)
        minimize = optimizer.minimize(loss)

        self.state_in = state_in
        self.h_in = h_in
        self.c_in = c_in
        self.action = action
        self.reward = reward
        self.prob = prob
        self.next_h = next_h
        self.next_c = next_c
        self.minimize = minimize
        self.loss = loss

        self.init_op = tf.global_variables_initializer()
        self.parameters.sess.run(self.init_op)

    def reset_rnn_state(self): #o.34 from inti
        self.h = np.zeros([self.parameters.batch_size, self.parameters.generator_H]) # gives matrix of zero's BxH
        self.c = np.zeros([self.parameters.batch_size, self.parameters.generator_H])

    def set_rnn_state(self, h, c):
        self.h = h
        self.c = c

    def get_rnn_state(self):
        return self.h, self.c

    def predict(self, state, stateful=True):
        # state = state.reshape(-1, 1)
        feed_dict = {
            self.state_in : state,
            self.h_in : self.h,
            self.c_in : self.c}
        prob, next_h, next_c = self.parameters.sess.run(
            [self.prob, self.next_h, self.next_c],
            feed_dict)

        if stateful:
            self.h = next_h
            self.c = next_c
            return prob
        else:
            return prob, next_h, next_c

    def update(self, state, action, reward,is_episode_end,h=None, c=None, stateful=True):
        #Update weights by Policy Gradient.
        if h is None:
            h = self.h
        if c is None:
            c = self.c
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1)
        feed_dict = {
            self.state_in : state,
            self.h_in : h,
            self.c_in : c,
            self.action : to_categorical(action, self.parameters.word_count),
            self.reward : reward}
        _, loss, next_h, next_c = self.parameters.sess.run(
            [self.minimize, self.loss, self.next_h, self.next_c],
            feed_dict)
        if stateful:
            self.h = next_h
            self.c = next_c
            return loss
        else:
            return loss, next_h, next_c

    def sampling_word(self, prob): #o.52 from below
        action = np.zeros((self.parameters.batch_size,), dtype=np.int32) #vector of zeros len B=32
        for i in range(self.parameters.batch_size): # for B times
            p = prob[i]   #o.53 get each row in the predicted matrix
            action[i] = np.random.choice(self.parameters.word_count, p=p) #taking randomly amount high probabilities an action
        return action

    def sampling_sentence(self, T, BOS=1):  #o.49 from below to get sentenses
        self.reset_rnn_state() #gives matrix of zero's BxH
        action = np.zeros([self.parameters.batch_size, 1], dtype=np.int32)  #o.50 gives vector of zeros shape(32,1)
        action[:, 0] = BOS #o.51 vector now all 1 flag for beggining of sentence
        actions = action
        for _ in range(T): #you prepering here 1 betch, each row is sentencse each colum is a word, in each loop
                           #you preper for each row an 1 action untill you reach T words for each row
            prob = self.predict(action) #o.52 predit for the corrent action the next (action)word => shepe=(B,V) each row contain all the words V and their probability for the current action
            action = self.sampling_word(prob).reshape(-1, 1) #o.53 take randomly high probabilities an action for each row
            actions = np.concatenate([actions, action], axis=-1) #o.54 attach to prev one the results
        # Remove BOS
        actions = actions[:, 1:]
        self.reset_rnn_state()
        return actions  #return betch shape = (B, T)

    def generate_samples(self, parameters):  #o.48 from pre-train D
        self.parameters.strT.print_to_TextBox(self.parameters.END, "Generating fake sentences \n")
        sentences=[]
        for _ in range(parameters.generate_samples // parameters.batch_size + 1):
            actions = self.sampling_sentence(parameters.seq_len)  #o.49 get betch shape = (B, T) row sentense colume words
            actions_list = actions.tolist() #o.54 return from making 1 batch
            for sentence_id in actions_list:  #o.55 just convert from id to word
                sentence = [parameters.id2word[action] for action in sentence_id] 
                sentences.append(sentence)
        output_str = ''
        for i in range(parameters.generate_samples):
            output_str += ' '.join(sentences[i]) + '\n'
        with open(parameters.path_neg, 'w', encoding='utf-8') as f: #o.55 save all in data as genereted samples
            f.write(output_str)

    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

def GeneratorPretrainingLSTM(parameters): #o.36 from train.init
    # in comment, B means batch size, T means lengths of time steps.
    input = Input(shape=(None,), dtype='int32', name='Input') # (B, T)
    out = Embedding(parameters.word_count, parameters.generator_E, mask_zero=True, name='Embedding')(input) # (B, T, E)
    out = LSTM(parameters.generator_H, return_sequences=True, name='LSTM')(out)  # (B, T, H)
    out = TimeDistributed(
        Dense(parameters.word_count, activation='softmax', name='DenseSoftmax'),
        name='TimeDenseSoftmax')(out)    # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining
