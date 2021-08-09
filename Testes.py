# É NECESSÁRIO UTILIZAR O PYTHON 3.7.0 NESSE CÓDIGO

import numpy as np
import cv2
from tkinter import *
from PIL import Image, ImageTk


cam = cv2.VideoCapture(0)

# loop infinito
while True:

    # ler imagem
    ret, frame = cam.read()
    dim = (1000, 600)
    imagem_dimensionada= cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 
    imagem_swp = cv2.cvtColor(imagem_dimensionada, cv2.COLOR_BGR2RGB)
    imagem_pil = Image.fromarray(imagem_swp)
    imagem_tk = ImageTk.PhotoImage(imagem_pil)


    # exibe a imagem
    cv2.imshow("RECONHECIMENTO FACIAL", frame)

    # aguarda alguma tecla
    key = cv2.waitKey(1)

    # para o sistema qando pressiona "q"
    if key == ord('q'):
        break

# libera a camera ou vídeo
cam.release()
# video.release()

# destrói as janelas do opencv
cv2.destroyAllWindows()








