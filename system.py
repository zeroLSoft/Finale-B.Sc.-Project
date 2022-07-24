import threading
from GUI.GUIDriver import GUIDriver, GUIState
from Input_init import TrainingInput
from eval_bleu import Evaluation
from tkinter import *
import random
import linecache
import re


class System:
    def __init__(self):
        self.gui_driver = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run_gui_handler).start()
        self.model = None

    def _run_gui_handler(self):
        self.lock.acquire()
        self.gui_driver = GUIDriver()
        self.gui_driver.run(self.lock)
        self.lock.release()

    def run(self):
        self.lock.acquire()
        self.lock.release()

        while True:
            while self.gui_driver.get_state() == GUIState.INIT:
                pass

            if self.gui_driver.get_state() == GUIState.CHANGE_FRAME:
                self.gui_driver.next_page()

            elif self.gui_driver.get_state() == GUIState.TRAINING:
                params_list = self.gui_driver.get_params()
                params_list[7] = self.gui_driver.get_text_widget()
                self.model = TrainingInput(params_list)
                self.model.trainFunc()
                self.gui_driver.set_trained_state()

            elif self.gui_driver.get_state() == GUIState.EVALUATION:
                eval = Evaluation(self.model, self.gui_driver.get_text_widget())
                eval.BLEU_test()

            elif self.gui_driver.get_state() == GUIState.GENERATION:
                widget = self.gui_driver.get_text_widget()
                for i in range(0, 20):
                    idx = random.randint(0, 10000)
                    sentence = linecache.getline(self.model.trainer.parameters.path_neg, idx)
                    stopwords = {'<PAD>', '<UNK>', '<S>', '</S>'}
                    resultwords = [word for word in re.split("\W+", sentence) if word not in stopwords]
                    result = ' '.join(resultwords)
                    stopwords = {'PAD', 'UNK', 'S', '/S'}
                    resultwords = [word for word in re.split("\W+", result) if word not in stopwords]
                    result = ' '.join(resultwords)
                    widget.print_to_TextBox(END, str(i + 1) + "." + result + "\n")

            #elif self.gui_driver.get_state() == GUIState.END_PROGRAM:
                #self.gui_driver.gui_driver.after(10, self.gui_driver.gui_driver.destroy())

            self.gui_driver.set_init_state()
