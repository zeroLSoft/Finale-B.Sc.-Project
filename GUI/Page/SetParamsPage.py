from tkinter import *
from GUI import GUIDriver
from tkinter import messagebox
from GUI.Page.Page import Page
from GUI.Page.Elements import Button, Label, SpinBox
from tkinter.filedialog import askdirectory, askopenfilename


class SetParamsPage(Page):
    def __init__(self, gui_driver, width, height, background):
        super().__init__(gui_driver=gui_driver, width=width, height=height, background=background)
        self.spinbox_init()

    def init_personal_params(self):
        self.params = []
        self.discCNN = 0
        self.discLSTM = 0
        self.genGRU = 0
        self.genLSTM = 0
        self.file = ""
        self.SaveLocation = ""
        self.preTrainEpochs = 0
        self.GANepochs = 0

    def spinbox_init(self):
        self.epochs = SpinBox.SpinBox(self.frame, scrollFrom=1, scrollTo=1000, increment=50, font=("Calibri", 20), width=10, background="white", x=300, y=440)
        self.gan_epochs = SpinBox.SpinBox(self.frame, scrollFrom=1, scrollTo=1000, increment=50, font=("Calibri", 20), width=10, background="white", x=300, y=520)

    def label_init(self):
        self.upload_instruction = Label.Label(self.frame, text="Upload Text File:", font=("Calibri Light", 20), background="#c3e1f7", x=20, y=120)
        self.save_instruction = Label.Label(self.frame, text="Saving Location:", font=("Calibri Light", 20), background="#c3e1f7", x=20, y=200)
        self.discriminator_select = Label.Label(self.frame, text="Discriminator Model:", font=("Calibri Light", 20), background="#c3e1f7", x=20, y=280)
        self.generator_select = Label.Label(self.frame, text="Generator Model:", font=("Calibri Light", 20), background="#c3e1f7", x=20, y=360)
        self.numOfepochs = Label.Label(self.frame, text="Pretrain Epochs:", font=("Calibri Light", 20), background="#c3e1f7", x=20, y=440)
        self.Ganepochs = Label.Label(self.frame, text="GAN Epochs:", font=("Calibri Light", 20), background="#c3e1f7", x=20, y=520)
        self.title_text = Label.Label(self.frame, text="Long Text Generation", font=("Berlin Sans FB Demi", 40), background="#c3e1f7", foreground="#0725e8", x=100, y=20)

    def button_init(self):
        self.nextTrain_btn = Button.Button(self.frame, command=lambda: self.trainingPageFunc(), text="Next", font="Calibri", height=1, width=7, background="white", foreground="black", x=550, y=650)
        self.browse_btn = Button.Button(self.frame, command=lambda: self.open_file(), text="Browse", font="Calibri", height=1, width=15, background="white", foreground="black", x=300, y=120)
        self.save_btn = Button.Button(self.frame, command=lambda: self.save(), text="Browse", font="Calibri", height=1, width=15, background="white", foreground="black", x=300, y=200)
        self.disCNN_btn = Button.Button(self.frame, command=lambda: self.disCNN(), text="CNN", font="Calibri", height=1, width=15, background="white", foreground="black", x=300, y=280)
        self.disLSTM_btn = Button.Button(self.frame, command=lambda: self.disLstm(), text="LSTM", font="Calibri", height=1, width=15, background="white", foreground="black", x=480, y=280)
        self.grGRU_btn = Button.Button(self.frame, command=lambda: self.grGRU(), text="GRU", font="Calibri", height=1, width=15, background="white", foreground="black", x=300, y=360)
        self.grLSTM_btn = Button.Button(self.frame, command=lambda: self.grLstm(), text="LSTM", font="Calibri", height=1, width=15, background="white", foreground="black", x=480, y=360)
        self.ep_btn = Button.Button(self.frame, command=lambda: self.preTrainepochsFunc(), text="Submit", font="Calibri", height=1, width=15, background="white", foreground="black", x=480, y=437)
        self.ganEp_btn = Button.Button(self.frame, command=lambda: self.GANepochsFunc(), text="Submit", font="Calibri", height=1, width=15, background="white", foreground="black", x=480, y=517)

    def open_frame(self):
        self.frame.tkraise()

    def get_parameters(self):
        return self.params

    def trainingPageFunc(self):
        if (self.discCNN == 1 or self.discLSTM == 1) and (self.genGRU == 1 or self.genLSTM == 1) and self.SaveLocation != "" and self.preTrainEpochs != 0 and self.GANepochs != 0 and self.file != "":
            self.params.clear()
            self.params = [
                self.discCNN,
                self.genLSTM,
                self.file,
                self.SaveLocation,
                self.preTrainEpochs,
                self.GANepochs,
                END,
                None
            ]
            self.state = GUIDriver.GUIState.CHANGE_FRAME
            self.nextPage = GUIDriver.GUIPages.TRAINING_PAGE
        else:
            messagebox.showerror("error", "Not all options are chosen")

    def open_file(self):
        if self.file != "":
            self.file = ""
        self.file = self.file + askopenfilename(parent=self.frame, title="choose a file", filetype=[("txt file", "*.txt")])
        if self.file != "":
            self.browse_btn.Edit_Button(background="#0725e8", foreground="white")

    def save(self):
        global SaveLocation
        if self.SaveLocation != "":
            self.SaveLocation = ""
        self.SaveLocation = self.SaveLocation + askdirectory(parent=self.frame, title="choose a location")
        if self.SaveLocation != "":
            self.save_btn.Edit_Button(background="#0725e8", foreground="white")

    def disCNN(self):
        self.discCNN = 1
        self.disCNN_btn.Edit_Button(background="#0725e8", foreground="white")
        self.discLSTM = 0
        self.disLSTM_btn.Edit_Button(background="white", foreground="black")

    def disLstm(self):
        self.discLSTM = 1
        self.disLSTM_btn.Edit_Button(background="#0725e8", foreground="white")
        self.discCNN = 0
        self.disCNN_btn.Edit_Button(background="white", foreground="black")

    def grGRU(self):
        self.genGRU = 1
        self.grGRU_btn.Edit_Button(background="#0725e8", foreground="white")
        self.genLSTM = 0
        self.grLSTM_btn.Edit_Button(background="white", foreground="black")

    # LSTM generator button
    def grLstm(self):
        self.genLSTM = 1
        self.grLSTM_btn.Edit_Button(background="#0725e8", foreground="white")
        self.genGRU = 0
        self.grGRU_btn.Edit_Button(background="white", foreground="black")

    # preTrain Epochs
    def preTrainepochsFunc(self):
        self.preTrainEpochs = self.epochs.get_Epoch()
        self.ep_btn.Edit_Button(background="#0725e8", foreground="white")

    # GAN Epochs
    def GANepochsFunc(self):
        self.GANepochs = self.gan_epochs.get_Epoch()
        self.ganEp_btn.Edit_Button(background="#0725e8", foreground="white")
