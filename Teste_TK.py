#coding: utf8
import sys
from tkinter import *
from PIL import Image, ImageTk 


def sair():
    sys.exit()


janela = Tk()
janela.attributes("-fullscreen", True)

janela.title("Teste TK")



tkimage = ImageTk.PhotoImage(Image.open("TEST.png"))
imagem = Label( text = "adicionando", image = tkimage)
imagem.image = tkimage
imagem.pack(padx=20, pady=20)



bt = Button(janela, width=20, text="Fechar", command = sair)

bt.pack(side=BOTTOM, padx=20, pady=20)

janela.geometry("800x700+290+0")


janela.mainloop()


