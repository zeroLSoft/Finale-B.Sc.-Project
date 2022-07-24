from GUI.Page.Elements import Button, Label, TextBox
from GUI.Page.Page import Page
from GUI import GUIDriver
from tkinter import messagebox


class TrainingPage(Page):
    def __init__(self, gui_driver, width, height, background):
        super().__init__(gui_driver=gui_driver, width=width, height=height, background=background)
        self.textBox_init()

    def init_personal_params(self):
        self.trained = False
        self.textBox_init()

    def button_init(self):
        self.train_btn = Button.Button(self.frame, command=lambda: self.startTrain(), text="Start training", font="Calibri", height=1, width=15, background="white", foreground="black", x=250, y=500)
        self.backToTrain_btn = Button.Button(self.frame, command=lambda: None, text="Back", font="Calibri", height=1, width=5, background="white", foreground="black", x=50, y=650)
        self.nextGen_btn = Button.Button(self.frame, command=lambda: self.open_generation(), text="Text generation", font="Calibri", height=1, width=15, background="white", foreground="black", x=550,
                                         y=650)
        self.nextEva_btn = Button.Button(self.frame, command=lambda: self.open_evaluation(), text="Show Evaluation", font="Calibri", height=1, width=15, background="white", foreground="black", x=250,
                                         y=550)

    def label_init(self):
        self.train_text = Label.Label(self.frame, text="Training procces", font=("Berlin Sans FB Demi", 22), background="#c3e1f7", foreground="#0725e8", x=235, y=100)

    def textBox_init(self):
        self.text_widget = TextBox.TextBox(self.frame, width=75, height=20, background="white", foreground="black", x=80, y=150)

    def startTrain(self):
        self.state = GUIDriver.GUIState.TRAINING

    def open_frame(self):
        self.frame.tkraise()

    def open_evaluation(self):
        self.open_New_Window(True)

    def open_generation(self):
        self.open_New_Window(False)

    def open_New_Window(self, flag):
        if (self.state == GUIDriver.GUIState.TRAINING):
            self.warning_handler("training")
        elif (not self.trained):
            self.warning_handler("not trained")
        else:
            self.state = GUIDriver.GUIState.CHANGE_FRAME
            if (flag):
                self.nextPage = GUIDriver.GUIPages.EVALUATION_PAGE
            else:
                self.nextPage = GUIDriver.GUIPages.GENERATION_PAGE

    def get_text_widget(self):
        return self.text_widget

    def warning_handler(self, str):
        if (str == "training"):
            messagebox.showerror("error", "Model still training, please wait")
        elif (str == "not trained"):
            messagebox.showerror("error", "Model NOT trained, please train first")

    def return_button(self):
        self.state = GUIDriver.GUIState.CHANGE_FRAME
        self.nextPage = GUIDriver.GUIPages.SET_PARAMETERS_PAGE
