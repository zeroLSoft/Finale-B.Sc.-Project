from GeneretorGRU import GeneratorGRU
from GeneretorLSTM import GeneratorLSTM
import numpy as np

class Agent(object):

    def __init__(self, parameters): #o.30 coming from train.trainer.init
        self.parameters=parameters
        if self.parameters.geneT==1:
           self.generator =GeneratorLSTM(parameters) #o.37 the actuall pretrained generetor model
        else:
           self.generator =GeneratorGRU(parameters)
        
    def act(self, state, epsilon=0, deterministic=False):  #o.74
        word = state[:, -1].reshape([-1, 1]) # o.75 we get state.shape(32,1-25)
        return self._act_on_word(word, epsilon=epsilon, deterministic=deterministic)

    def _act_on_word(self, word, epsilon=0, deterministic=False, PAD=0, EOS=2): #o.72
        action = None
        is_PAD = word == PAD
        is_EOS = word == EOS
        is_end = is_PAD.astype(np.int) + is_EOS.astype(np.int)
        is_end = 1 - is_end
        is_end = is_end.reshape([self.parameters.batch_size, 1])
        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=self.parameters.word_count, size=(self.parameters.batch_size, 1))
        elif not deterministic:
            probs = self.generator.predict(word)  #o.76 for the current word predict probabilities for the next one
            action = self.generator.sampling_word(probs).reshape([self.parameters.batch_size, 1])  #o.77 here we already used this func, its getting us randome worf of the best probs
        else:
            probs = self.generator.predict(word) # (B, T)
            action = np.argmax(probs, axis=-1).reshape([self.parameters.batch_size, 1])
        return action * is_end

    def reset(self):
        self.generator.reset_rnn_state()

    def save(self, path):
        self.generator.save(path)

    def load(self, path):
        self.generator.load(path)


class Environment(object):
    def __init__(self, discriminator, g_beta, parameters): #o.36 from train.init
        self.parameters=parameters
        self.discriminator = discriminator  #the discriminator
        self.g_beta = g_beta # the agent, the second one
        self.reset()

    def get_state(self):   #o.65 
        if self.t == 1:    #o.66 if its the beggining
            return self._state
        else:
            return self._state[:, 1:]   # Exclude BOS

    def reset(self):
        self.t = 1
        self._state = np.zeros([self.parameters.batch_size, 1], dtype=np.int32)
        self._state[:, 0] = self.parameters.BOS
        self.g_beta.reset()

    def step(self, action): #o.66 we get action which is current state at time t

        self.t = self.t + 1 #o.67 update t to know what action we are at

        reward = self.Q(action, self.parameters.MCS_sample)  #o.68 get reword shape(32.1) rewords for each state, sampled in rollout
        is_episode_end = self.t > self.parameters.seq_len  #o.78 see if ended

        self._append_state(action) #o.78 see if ended
        next_state = self.get_state() #o.80 get current state
        info = None

        return [next_state, reward, is_episode_end, info]

    def render(self, head=1):
        for i in range(head):
            ids = self.get_state()[i]
            words = [self.parameters.id2word[id] for id in ids.tolist()]
            print(' '.join(words))
        print('-' * 80)


    def Q(self, action, n_sample=16): #o.68

        if self.parameters.geneT==1:
           h, c = self.g_beta.generator.get_rnn_state()
        else:
            h= self.g_beta.generator.get_rnn_state() #o.69 this is the other agant, num2, we get his hiddin state
        reward = np.zeros([self.parameters.batch_size, 1]) #o.70 just prep matrix
        if self.t == 2:
            Y_base = self._state    # Initial case
        else:
            Y_base = self.get_state()    # (B, t-1)

        if self.t >= self.parameters.seq_len+1: #o.71 its when we complete the state until T
            Y = self._append_state(action, state=Y_base)
            return self.discriminator.predict(Y)

        # Rollout
        for idx_sample in range(n_sample): #o.72 for 16 times sample
            Y = Y_base  # build of state Y starting from (32,1)
            if self.parameters.geneT==1:
               self.g_beta.generator.set_rnn_state(h, c)
            else:
               self.g_beta.generator.set_rnn_state(h)  #o.73
           
            y_t = self.g_beta.act(Y, epsilon=self.parameters.eps)  #o.74 get actions for current Y 
            Y = self._append_state(y_t, state=Y)    #o.75 build full state
            for tau in range(self.t+1, self.parameters.seq_len): #o.76 sampele the rest of the state
                y_tau = self.g_beta.act(Y, epsilon=self.parameters.eps)
                Y = self._append_state(y_tau, state=Y)
            reward += self.discriminator.predict(Y) / n_sample  #o.77 and the for each state (vector (32,1) get reword)

        return reward


    def _append_state(self, word, state=None):
        word = word.reshape(-1, 1)
        if state is None:
            self._state = np.concatenate([self._state, word], axis=-1)
        else:
            return np.concatenate([state, word], axis= -1)
