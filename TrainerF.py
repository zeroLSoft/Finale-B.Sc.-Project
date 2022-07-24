from Data_manager import GeneratorDataManeger, DiscriminatorDataManager
from DiscriminatorCNN import DiscriminatorCNN
from DiscriminatorLSTM import DiscriminatorLSTM
from GeneretorGRU import GeneratorPretrainingGRU
from GeneretorLSTM import GeneratorPretrainingLSTM
from reinforcement_learning import Agent, Environment
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

sess = tf.Session()
import keras.backend as K

K.set_session(sess)
from keras import backend as K

K.tensorflow_backend.set_session(sess)
from Header import Parameters
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class Trainer(object):
    def __init__(self, discriT, geneT, locations, lines, strT, END):
        self.parameters = Parameters(discriT, geneT, locations, sess, strT, END, lines)
        self.parameters.strT.print_to_TextBox(self.parameters.END, "Initializing parameters...\n")
        self.parameters.g_data = GeneratorDataManeger(self.parameters)  # o.11 prepere pretrain generator and data in utility
        self.parameters.d_data = DiscriminatorDataManager(self.parameters)
        self.agent = Agent(self.parameters)  # o.30 init agent in RL, sess is Session in TensorFlow.
        self.g_beta = Agent(self.parameters)  # o.34 return from agant, same process as o.30

        if discriT == 1:
            self.discriminator = DiscriminatorCNN(self.parameters)
        else:
            self.discriminator = DiscriminatorLSTM(self.parameters)  # o.35 init Discriminator

        self.env = Environment(self.discriminator, self.g_beta, self.parameters)  # o.36 prep the inv
        if geneT == 1:
            self.generator_pre = GeneratorPretrainingLSTM(self.parameters)  # o.37 the actuall pretrained generetor model
        else:
            self.generator_pre = GeneratorPretrainingGRU(self.parameters)

    def pre_train_generator(self, g_epochs=3):  # o.39 pretrain generetor
        lr = 1e-2
        g_adam = Adam(lr)  # o.41 init optimizer
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')  # o.42 compile, the generator_pre from o.37
        self.parameters.strT.print_to_TextBox(self.parameters.END, "Generator pre-training in progress...\n")
        self.generator_pre.summary()  # p.43 show model

        # o.44 train the model
        self.generator_pre.fit_generator(self.parameters.g_data, steps_per_epoch=None, epochs=g_epochs)
        self.generator_pre.save_weights(self.parameters.pretrain_generatorL)  # o.45 save pretrain weights to the file path
        self.reflect_pre_train()  # o.46, set weight to the agent
        self.parameters.strT.print_to_TextBox(self.parameters.END, "Generator pre-training Done!\n")

    def pre_train_discriminator(self):  # o.47 pretrain discriminator from o.35
        d_epochs = 1
        lr = 1e-4
        self.parameters.strT.print_to_TextBox(self.parameters.END, "Discriminator pre-training in progress...\n")
        self.agent.generator.generate_samples(self.parameters)
        self.d_data = DiscriminatorDataManager(self.parameters)  # o.55 return after generating sampels
        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()

        self.discriminator.fit_generator(self.d_data, steps_per_epoch=None, epochs=d_epochs)
        self.discriminator.save(self.parameters.pretrain_discriminatorL)  # o.57 save, save is outside func
        self.parameters.strT.print_to_TextBox(self.parameters.END, "Discriminator pre-training Done!\n")

    def load_pre_train(self):  # o.58 lead the pretrainn data
        self.generator_pre.load_weights(self.parameters.pretrain_generatorL)  # load_weights iss outsise func
        self.reflect_pre_train()  # load_weights for G
        self.discriminator.load_weights(self.parameters.pretrain_discriminatorL)  # load_weights iss outsise func

    def load_pre_train_g(self):
        self.generator_pre.load_weights(self.parameters.pretrain_generatorL)
        self.reflect_pre_train()

    def load_pre_train_d(self):
        self.discriminator.load_weights(self.parameters.pretrain_discriminatorL)

    def reflect_pre_train(self):  # o.46 from pretrain
        i = 0
        for layer in self.generator_pre.layers:  # generator_pre is a model, so layers is outside func
            if len(layer.get_weights()) != 0:  # get layer from pretrained model and see if empty
                w = layer.get_weights()  # get the get_weights
                self.agent.generator.layers[i].set_weights(w)  # update agent
                self.g_beta.generator.layers[i].set_weights(w)  # and second agent
                i += 1

    def train(self, steps=10, g_steps=1, d_steps=1, d_epochs=1, verbose=True, head=1):  # o.61 start training from main
        d_adam = Adam(self.parameters.discriminator_lr)
        self.parameters.strT.print_to_TextBox(self.parameters.END, "\nGAN training initialized \n")
        self.discriminator.compile(d_adam, 'binary_crossentropy')  # o.62 compile the discriminator as always
        self.eps = self.parameters.eps
        j = 0
        for step in range(steps):
            self.parameters.strT.print_to_TextBox(self.parameters.END, "GAN step: " + str(step + 1) + "/" + str(steps) + "\n")
            self.parameters.strT.print_to_TextBox(self.parameters.END, "Generator is training \n")
            for _ in range(g_steps):
                rewards = np.zeros([self.parameters.batch_size, self.parameters.seq_len])  #
                self.agent.reset()  # o.63 start from zero
                self.env.reset()  #
                self.parameters.strT.print_to_TextBox(self.parameters.END, "Building state and update generator in each step \n")
                for t in range(self.parameters.seq_len):  # o.64 for T size sentence len build state
                    self.parameters.strT.print_to_TextBox(self.parameters.END, "state in: " + str(t + 1) + "/" + str(self.parameters.seq_len) + "\n")
                    state = self.env.get_state()  # o.65 get state: t1: i, t2: i study, t3 i study deep.... state.shape()
                    # =(32,25) at t=T time, we besicly build betch
                    action = self.agent.act(state, self.eps)  # o.66 for the current state get actine
                    next_state, reward, is_episode_end, info = self.env.step(action)  # o.80 get t+1 state and his rewords sampeled in rollout
                    self.agent.generator.update(state, action, reward, is_episode_end)
                    rewards[:, t] = reward.reshape([self.parameters.batch_size, ])
                    if is_episode_end:
                        if verbose:
                            avg = np.average(rewards)
                            self.parameters.strT.print_to_TextBox(self.parameters.END, "Reword for the whole state: " + str(avg) + "\n")
                            self.parameters.training_datas.append((j, avg))
                        break
            temp = []
            for _ in range(d_steps):
                self.agent.generator.generate_samples(self.parameters)
                self.parameters.strT.print_to_TextBox(self.parameters.END, "Discriminator is training... \n")
                self.parameters.d_data = DiscriminatorDataManager(self.parameters)
                self.discriminator.fit_generator(self.parameters.d_data, steps_per_epoch=None, epochs=d_epochs)

            j += 1
            # Update env.g_beta to agent
            self.agent.save(self.parameters.generatorL)
            self.g_beta.load(self.parameters.generatorL)

            # self.discriminator.save(self.parameters.discriminatorL)
            self.eps = max(self.eps * (1 - float(step) / steps * 4), 1e-4)

            self.save()
            self.load()

        xx, yy = zip(*self.parameters.training_datas)
        plt.figure(1)
        plt.title('Rewords')
        plt.xlabel('Epochs')
        plt.ylabel('reword')
        plt.plot(xx, yy)
        plt.show()

    def save(self):
        self.agent.save(self.parameters.generatorL)
        self.discriminator.save(self.parameters.discriminatorL)

    def load(self):
        self.agent.load(self.parameters.generatorL)
        self.g_beta.load(self.parameters.generatorL)
        self.discriminator.load_weights(self.parameters.discriminatorL)
