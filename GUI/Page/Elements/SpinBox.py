import tkinter as tk


class SpinBox:
    def __init__(self, frame, scrollFrom=1, scrollTo=1000, increment=50, font=("Calibri", 20), width=10, background="white",x=300, y=440):
        self.SpinBox = tk.Spinbox(frame, from_=scrollFrom, to=scrollTo, increment=increment, font=font, width=width, bg=background)
        self.SpinBox.place(x=x, y=y)
        self.get_SpinBox()

    def get_SpinBox(self):
        return self.SpinBox

    def get_Epoch(self):
        return self.SpinBox.get()