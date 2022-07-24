import nltk
import re

class Evaluation():
    def __init__(self, model, strE):
        self.strE = strE
        self.END = model.trainer.parameters.END
        self.strE.print_to_TextBox(self.END, "initializing parameters \n")
        self.model = model
        self.model.trainer.agent.generator.generate_samples(self.model.trainer.parameters)
        self.reference = []
        self.hypothesis_list = []
        self.strE.print_to_TextBox(self.END, "initializing candidates \n")

        for sentence in self.model.trainer.parameters.sentences:
            candidate = self.model.trainer.parameters.vocab.sentence_to_ids(sentence)  # list of ids
            self.reference.append(candidate)

        self.strE.print_to_TextBox(self.END, "initializing references \n")
        with open(self.model.trainer.parameters.path_neg) as fin:
            for line in fin:
                stopwords = {'<PAD>', '<UNK>', '<S>', '</S>'}
                resultwords = [word for word in re.split("\W+", line) if word not in stopwords]
                result = ' '.join(resultwords)
                stopwords = {'PAD', 'UNK', 'S', '/S'}
                resultwords = [word for word in re.split("\W+", result) if word not in stopwords]
                print(resultwords)
                self.hypothesis_list.append(self.model.trainer.parameters.vocab.sentence_to_ids(resultwords))

    def BLEU_test(self):
        self.strE.print_to_TextBox(self.END, "BLEU_test start... \n")
        for ngram in range(2, 5):
            weight = tuple((1. / ngram for _ in range(ngram)))
            bleu = []
            num = 0
            for h in self.hypothesis_list[:1000]:
                BLEUscore = nltk.translate.bleu_score.sentence_bleu(self.reference, h, weight)
                if (num % 100 == 0):
                    self.strE.print_to_TextBox(self.END, str(len(weight)) + "-gram: 1000/" + str(num) + "\n")
                num += 1
                bleu.append(BLEUscore)
            self.strE.print_to_TextBox(self.END, str(len(weight)) + "-gram BLEU score : " + str(1.0 * sum(bleu) / len(bleu)) + "\n")
