import tkinter as tk


class Button:
    def __init__(self, frame, command, text, font="Calibri", height=1, width=15, background="white",   foreground="black",x=None,y=None):
        self.Button=tk.Button(frame, command=command, text=text, font=font, height=height, width=width, bg=background, fg=foreground)
        self.Button.place(x=x, y=y)
        self.get_Button()

    def Edit_Button(self, background="#0725e8", foreground="white"):
        self.Button.configure(bg=background, fg=foreground)

    def get_Button(self):
        return self.Button
