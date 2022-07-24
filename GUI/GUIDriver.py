from dataclasses import dataclass
import tkinter as tk
from GUI.Page.SetParamsPage import SetParamsPage
from GUI.Page.TrainingPage import TrainingPage
from GUI.Page.EvaluationPage import EvaluationPage
from GUI.Page.GenerationPage import GenerationPage


@dataclass
class GUIState:
    INIT = 0
    SET_PARAMS = 1
    TRAINING = 2
    CHANGE_FRAME = 3
    EVALUATION = 4
    GENERATION = 5
    END_PROGRAM = 6


@dataclass
class GUIPages:
    SET_PARAMETERS_PAGE = 0
    TRAINING_PAGE = 1
    EVALUATION_PAGE = 2
    GENERATION_PAGE = 3


class GUIDriver:
    def __init__(self):
        self.gui_driver = tk.Tk()
        self.gui_driver.resizable(False, False)
        self.gui_driver.rowconfigure(0, weight=1)
        self.gui_driver.columnconfigure(0, weight=1)
        self.gui_driver.protocol("WM_DELETE_WINDOW", self.on_close)
        self.state = GUIState.INIT
        self.currentPage = GUIPages.SET_PARAMETERS_PAGE
        self.prevPage = None
        self.pages = [
            SetParamsPage(self.gui_driver, width=700, height=700, background="#c3e1f7"),
            TrainingPage(self.gui_driver, width=700, height=700, background="#c3e1f7"),
            EvaluationPage(self.gui_driver, width=700, height=700, background="#c3e1f7"),
            GenerationPage(self.gui_driver, width=700, height=700, background="#c3e1f7")
        ]

    def get_state(self):
        if (self.pages[self.currentPage].state != GUIState.INIT):
            self.state = self.pages[self.currentPage].state
            return self.state
        if self.state==GUIState.END_PROGRAM:
            return self.state
        return GUIState.INIT

    def next_page(self):
        self.prevPage = self.currentPage
        self.currentPage = self.pages[self.currentPage].nextPage
        self.pages[self.currentPage].open_frame()

    def run(self, lock):
        self.pages[self.currentPage].open_frame()
        lock.release()
        self.gui_driver.mainloop()

    def get_params(self):
        return self.pages[0].get_parameters()

    def get_text_widget(self):
        return self.pages[self.currentPage].get_text_widget()

    def set_init_state(self):
        self.state = GUIState.INIT
        self.pages[self.prevPage].nextPage = None
        self.pages[self.currentPage].state = GUIState.INIT

    def set_trained_state(self):
        self.pages[self.currentPage].trained = True

    def on_close(self):
        print("hrre")
        self.state = GUIState.END_PROGRAM
        print(self.state)
