# É NECESSÁRIO UTILIZAR O PYTHON 3.7.0 NESSE CÓDIGO

from fdlite import FaceDetection
from fdlite.render import Colors, detections_to_render_data, render_to_image
from PIL import Image
import numpy as np
import cv2


cam = cv2.VideoCapture(0)


detect_faces = FaceDetection()
# loop infinito
while True:

    # ler imagem
    ret, frame = cam.read()
    opencv_image = frame
    color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_coverted)
    
    faces = detect_faces(pil_image)


    if len(faces) == 0:
        print('no faces detected :(')
    else:
        render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
        img = render_to_image(render_data, pil_image)
        numpy_image = np.array(img)  
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 


    # exibe a imagem
    cv2.imshow("RECONHECIMENTO FACIAL", opencv_image)

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








