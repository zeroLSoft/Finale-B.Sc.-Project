import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dropout, Concatenate
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D


def DiscriminatorCNN(parameters):

    filter_sizes= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(parameters.word_count, parameters.discriminator_E, name='Embedding')(input)  # (B, T, E)
    out = VariousConv1D(out, filter_sizes, num_filters)
    out = Highway(out, num_layers=1)
    out = Dropout(parameters.dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator

def VariousConv1D(x, filter_sizes, num_filters):

    conv_outputs = []
    for filter_size, n_filter in zip(filter_sizes, num_filters):
        conv_out = Conv1D(n_filter, filter_size)(x)   # (B, time_steps, n_filter)
        conv_out = GlobalMaxPooling1D()(conv_out) # (B, n_filter)
        conv_outputs.append(conv_out)
    out = Concatenate()(conv_outputs)
    return out

def Highway(x, num_layers=1):

    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio = Dense(input_size, activation='sigmoid')(x)
        fc = Dense(input_size, activation='relu')(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]))([fc, x, gate_ratio])
    return x
