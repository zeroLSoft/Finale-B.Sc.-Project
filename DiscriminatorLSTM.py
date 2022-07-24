import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dropout
from keras.layers import Dense, Embedding, LSTM


def DiscriminatorLSTM(parameters):
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(parameters.word_count, parameters.discriminator_E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = LSTM(parameters.discriminator_H)(out)
    out = Highway(out, num_layers=1)
    out = Dropout(parameters.dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out) #output: probability of true data or not, shape = (B, 1)
    return discriminator

def Highway(x, num_layers=1):
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio = Dense(input_size, activation='sigmoid')(x)
        fc = Dense(input_size, activation='relu')(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]))([fc, x, gate_ratio])
    return x
