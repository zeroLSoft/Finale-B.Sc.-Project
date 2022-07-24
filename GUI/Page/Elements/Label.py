import tkinter as tk


class Label:
    def __init__(self, frame, text="", font=("Berlin Sans FB Demi", 22), background="#c3e1f7", foreground="#0725e8",x=0, y=0):
        self.Label = tk.Label(frame, text=text, font=font, bg=background, fg=foreground)
        self.Label.place(x=x, y=y)
        self.get_Label()

    def get_Label(self):
        return self.Label
