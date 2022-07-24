from abc import ABC
import tkinter as tk
from GUI import GUIDriver

class Page(ABC):
    def __init__(self, gui_driver, width, height, background):
        self.gui_driver = gui_driver
        self.frame = tk.Frame(self.gui_driver, width=width, height=height, bg=background)
        self.frame.grid(row=0, column=0, sticky='nsew')
        self.state = GUIDriver.GUIState.INIT
        self.nextPage = None

        self.init_personal_params()
        self.label_init()
        self.button_init()

    def init_personal_params(self):
        pass

    def on_close(self):
        pass

    def spinbox_init(self):
        pass

    def label_init(self):
        pass

    def button_init(self):
        pass
