import tkinter as tk


class TextBox:
    def __init__(self, frame, width=75, height=20, background="white", foreground="black", x=0, y=0):
        self.TextBox = tk.Text(frame, width=width, height=height, bg=background, fg=foreground)
        self.TextBox.place(x=x, y=y)
        self.get_TextBox()

    def get_TextBox(self):
        return self.TextBox

    def print_to_TextBox(self, END, str):
        self.TextBox.insert(END, str)
        self.TextBox.see("end")
