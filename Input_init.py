from TrainerF import Trainer


class TrainingInput:
    def __init__(self, parameters):
        self.trainer = None
        self.locations = []
        self.strT = parameters[7]
        self.END = parameters[6]
        self.location = parameters[3]
        self.locations.append(self.location + '\generator_pre.hdf5')
        self.locations.append(self.location + '\discriminator_pre.hdf5')
        self.locations.append(self.location + '\generator.pkl')
        self.locations.append(self.location + '\discriminator.hdf5')
        self.locations.append(self.location + '\generated_sentences.txt')

        if parameters[0] == 1:  # choice of discriminator model
            self.val1 = 1
        else:
            self.val1 = 2

        if parameters[1] == 1:  # choice of generator model
            self.val2 = 1
        else:
            self.val2 = 2
        self.val3 = parameters[4]  # pretrained epochs
        self.val4 = parameters[5]  # epochs
        self.location2 = parameters[2]
        open(self.locations[4], 'a').close()

        self.count = 0
        with open(self.location2, 'r', encoding='utf-8') as f:
            for line in f:
                if (len(line.split()) > 0):
                    self.count += 1

        self.locations.append(self.location2)
        self.locations.append(self.location + '\original_sentences_id.txt')
        self.locations.append(self.location + '\generated_sentences_id.txt')

    def trainFunc(self):
        self.strT.print_to_TextBox(self.END, "Training start\n")
        self.trainer = Trainer(self.val1, self.val2, self.locations, self.count, self.strT, self.END)
        self.trainer.pre_train_generator(int(self.val3))
        self.trainer.pre_train_discriminator()
        self.trainer.load_pre_train()  # o.57 return after finishing pretrain discriminator
        self.trainer.reflect_pre_train()  # o.59 again init weight
        self.trainer.train(steps=int(self.val4), g_steps=1, head=10)  # o.60 start training
        self.strT.print_to_TextBox(self.END, "Training done!\n")

