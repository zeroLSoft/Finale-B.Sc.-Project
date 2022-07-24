from Data_manager import Vocab,load_data

class Parameters(object):
    def __init__(self, discriT,geneT,locations,sess,strT,END,lines=None):
        self.sess=sess
        self.discriT=discriT #flag to know CNN or LSTM
        self.geneT=geneT     #flag to know LSTM or GRU
        self.PAD,self.BOS ,self.EOS, self.UNK = 0,1,2,3 #sentence fillers and flags
        self.PAD_TOKEN = '<PAD>'   #space filler, len=8, sentence=My name is guotong1988 =>My name is guotong1988 _pad_ _pad_ _pad_ _pad_
        self.UNK_TOKEN = '<UNK>'   #unknown token
        self.BOS_TOKEN = '<S>'     #beggining of sentence
        self.EOS_TOKEN = '</S>'    #end of sentence
        self.pretrain_generatorL=locations[0]      #generator pretrain weights location
        self.pretrain_discriminatorL=locations[1]  #discriminator pretrain weights location
        self.generatorL=locations[2]               #generator weights location
        self.discriminatorL=locations[3]           #discriminator weights location
        self.path_pos = locations[5]     #positive path location
        self.path_neg = locations[4]     #negative path location
        self.path_pos_id = locations[6]
        self.path_neg_id = locations[7]
        self.batch_size = 32     #32 batch size
        self.seq_len = 25        #25 sentence len
        self.min_count = 1
        self.generator_E=64
        self.generator_H=64
        self.discriminator_E=64
        self.discriminator_H=64
        self.generator_lr= 1e-3
        self.discriminator_lr= 1e-3
        self.dropout = 0.1
        self.generate_samples = 10000
        self.MCS_sample=16
        self.eps = 0.1
        default_dict = {self.PAD_TOKEN: self.PAD,self.BOS_TOKEN: self.BOS,self.EOS_TOKEN: self.EOS,self.UNK_TOKEN: self.UNK,}
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)#o.14 init the class vocab
        self.sentences = load_data(self.path_pos) #o.18 return from Vocab and load the sentences from text
        self.vocab.build_vocab(self.sentences, self.min_count) #o.19 give all ward count and ids
        self.word2id = self.vocab.word2id #o.24 return from vocab.build_vocab and init ids to class this and 3 below
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab  #all words and their count
        self.word_count = len(self.vocab.word2id)    # how many words in text in total
        self.n_data=lines   #how many lines
        self.n_dataD=0
        self.shuffle = True  #FLAG set to true
        self.idxG = 0
        self.idxD = 0
        self.g_data=None
        self.d_data=None
        self.strT=strT
        self.END=END
        self.training_data = []
        self.training_datas = []
        self.epochF=0
