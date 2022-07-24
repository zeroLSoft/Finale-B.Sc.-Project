import tkinter as tk
from GUI.Page.Page import Page
from GUI.Page.Elements import Button, Label, TextBox
from GUI import GUIDriver


class EvaluationPage(Page):
    def __init__(self, gui_driver, width, height, background):
        super().__init__(gui_driver=gui_driver, width=width, height=height, background=background)

    def init_personal_params(self):
        self.textBox_init()

    def button_init(self):
        self.eva_btn = Button.Button(self.frame, command=lambda: self.startEval(), text="Start evaluation", font="Calibri", height=1, width=15, background="white", foreground="black", x=550, y=650)
        self.backToTrain_btn = Button.Button(self.frame, command=lambda: self.return_Button(), text="Back", font="Calibri", height=1, width=5, background="white", foreground="black", x=50, y=650)

    def label_init(self):
        self.eva_text = Label.Label(self.frame, text="Evaluation ", font=("Berlin Sans FB Demi", 22), background="#c3e1f7", foreground="#0725e8", x=265, y=100)

    def textBox_init(self):
        self.text_widget = TextBox.TextBox(self.frame, width=75, height=20, background="white", foreground="black", x=80, y=150)

    def startEval(self):
        self.state = GUIDriver.GUIState.EVALUATION

    def open_frame(self):
        self.frame.tkraise()

    def return_Button(self):
        self.state = GUIDriver.GUIState.CHANGE_FRAME
        self.nextPage = GUIDriver.GUIPages.TRAINING_PAGE

    def get_text_widget(self):
        return self.text_widget
