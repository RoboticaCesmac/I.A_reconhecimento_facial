# É NECESSÁRIO UTILIZAR O PYTHON 3.6.0 NESSE CÓDIGO

import numpy as np
from PIL import Image
from mtcnn_tflite.MTCNN import MTCNN
#from mtcnn.mtcnn import MTCNN
from tensorflow import lite
import cv2
from sklearn.preprocessing import Normalizer
import time
from firebase_admin import credentials
from firebase_admin import db
import firebase_admin


#credenciais firebase
cred = firebase_admin.credentials.Certificate("firebase.json")

#inicializa firebase
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://reconhecimento-facial-aa406-default-rtdb.firebaseio.com/'
})

#referência database
ref = db.reference()


# detector de faces
detector = MTCNN()

# extrator de embeddings
interpreterFacenet = lite.Interpreter(model_path="facenet_keras.tflite")
input_detailsFacenet = interpreterFacenet.get_input_details()
output_detailsFacenet = interpreterFacenet.get_output_details()
interpreterFacenet.allocate_tensors()

# modelo de reconhecimento de faces
interpreterFaces = lite.Interpreter(model_path="faces.tflite")
input_detailsFaces = interpreterFaces.get_input_details()
output_detailsFaces = interpreterFaces.get_output_details()
interpreterFaces.allocate_tensors()

# detector de máscaras
interpreterMask = lite.Interpreter(model_path="detector_mascara.tflite")
input_detailsMask = interpreterMask.get_input_details()
output_detailsMask = interpreterMask.get_output_details()
interpreterMask.allocate_tensors()


# classes da rede (nome das pessoas)
pessoa = [
    "Alvaro",
    "Artur",
    "Cristovao",
    "Fernando",
    "Guilherme",
    "Mozart",
    "Udson",
    "Desconhecido"
]

# número de classes / pessoas
num_classes = len(pessoa)

# captura de imagem (parâmetros: "nomeVídeo.mp4" ou 0 para webcam)
cam = cv2.VideoCapture(0)

#Contador do fps
start_time = time.time()
# FPS update time in seconds
display_time = 2
fc = 0
FPS = 0


# função extrair face
def extract_face(image, box,  required_size=(160, 160)):
    # coordenadas x, y, largura e altura
    x, y, w, h = box

    # essa condição serve para impedir que uma face cortada retorne uma coordenada negativa
    if (x > 0 and y > 0):

        # passagem de imagem para numpy array
        pixels = np.asarray(image)

        # coordenadas x, y, largura e altura
        x, y, w, h = box

        # coordenadas secundárias para fechar a caixa de detecção
        x2, y2 = x + w, y + h

        # extrai a face dos pixels com as coordenadas
        face = pixels[y: y2, x: x2]

        # extrai a imagem do array
        image = Image.fromarray(face)

        # redimensiona a imagem
        image = image.resize(required_size)

        # retorna a imagem como array
        return np.asarray(image)

    else:
        # retorna coordenadas negativas se ativado.
        return [[[0]]]


# Função para extrair embeddings (dados faciais)
def get_embeddig(face_pixels):

    # normalização
    face_pixels = face_pixels.astype("float32")

    # normalização
    mean, std = face_pixels.mean(), face_pixels.std()

    # normalização
    face_pixels = (face_pixels - mean) / std

    # expnsão dee dimensões para matriz
    samples = np.expand_dims(face_pixels, axis=0)

    # Extração da embedding
    samples2 = samples.astype(np.float32)       
    interpreterFacenet.set_tensor(input_detailsFacenet[0]['index'], samples2)
    interpreterFacenet.invoke()
    yhat = interpreterFacenet.get_tensor(output_detailsFacenet[0]['index'])

    # retorna embedding
    return yhat[0]



# loop infinito
while True:

    # ler imagem
    ret, frame = cam.read()



    #contador do fps
    fc+=1
    TIME = time.time() - start_time
    if (TIME) >= display_time :
	    FPS = fc / (TIME)
	    fc = 0
	    start_time = time.time()
    fps_disp = "FPS: "+str(FPS)[:5]
    # Add FPS count on frame
    frame = cv2.putText(frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)





    # detectar faces na imagem
    faces = detector.detect_faces(frame)

    # caso não tenha faces
    if(len(faces) == 0):
        ref.set({
            'face': "nada",
        })

    # para cada face detectada...
    for face in faces:

        # nível de confiança / probabilidade de ser uma face
        confidence = face['confidence']*100

        # alta probabilidade para evitar falso positivo (98%)
        if confidence >= 98:

            # coordenadas x, y, largura e altura
            x, y, w, h = face['box']
            # coordenada secundária x
            x2 = x + w
            # coordenada secundária y
            y2 = y + h

            # extrai a face
            face = extract_face(frame, face['box'])

            # antiCut permite evitar o erro de coordenada negativa
            antiCut = face[0]
            # antiCut permite evitar o erro de coordenada negativa
            antiCut = antiCut[0]
            # antiCut permite evitar o erro de coordenada negativa
            antiCut = antiCut[0]

            # Apenas continua se as coordenadas não forem negativas
            if (antiCut != 0):
                # extrai a região de interesse (face)
                roi = frame[y:y2, x:x2]

                # redimensiona a imagem
                roi = cv2.resize(roi, (160, 160))
                # redimensiona a imagem
                roi = np.reshape(roi, [1, 160, 160, 3])

                if np.sum([roi]) != 0:
                    # normalização
                    
                    roi = (roi.astype('float')/255.0)
                   
                    roi2 = roi.astype(np.float32)
                  

                    # verifica se está usando máscara
                    interpreterMask.set_tensor(input_detailsMask[0]['index'], roi2)
                    interpreterMask.invoke()
                    result = interpreterMask.get_tensor(output_detailsMask[0]['index'])

                    # extrai o resultado
                    result = result[0]

                    # se estiver sem máscara (probabilidade de 98% ou mais)
                    if result[0] >= 0.98:

                        # normalização
                        face = face.astype("float32")/255

                        # extrai embedding
                        embedding = get_embeddig(face)

                        # transforma em uma matriz (tensor)
                        tensor = np.expand_dims(embedding, axis=0)

                        # normalização l2
                        norm = Normalizer(norm="l2")

                        # executa normalização
                        tensor = norm.transform(tensor)


                        # prediz qual é a face detectada
                        tensor2 = tensor.astype(np.float32)
                        interpreterFaces.set_tensor(input_detailsFaces[0]['index'], tensor2)
                        interpreterFaces.invoke()
                        result = interpreterFaces.get_tensor(output_detailsFaces[0]['index'])
                
                        # recupera a probabilidade
                        prob = np.max(result)*100

                        # recupera a classe
                        classe  = np.argmax(result)

                        # alta probabilidade para reconhecimento facial
                        if prob >= 98:

                            # ajusta uma cor para pessoas desconhecidas
                            if classe == 7:
                                ref.set({
                                    'face': "desconhecido",
                                })
                                # BGR - VERMELHO
                                color = (0, 0, 255)

                            # ajusta uma cor para pessoas conhecidas
                            else:

                                ref.set({
                                    'face': "conhecido",
                                })

                                # BGR - VERDE
                                color = (0, 255, 0)

                            # Transforma o nome em caixa alta se necessário
                            user = str(pessoa[classe]).upper()

                            # extrai o nome da pessoa e a probabilidade
                            # user = user + " " + str(prob)

                            # extrai o centro da face para exibir um círculo
                            center = (int((x+w/2)), int(y+h/2))
                            # extrai o centro da face para exibir um círculo
                            y2 = y + h
                            # extrai o raio para exibir o círculo
                            radius = int((y2 - y) / 2)

                            # exibe uma caixa ao invés de um círculo
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # exibe um círculo na face detectada
                            cv2.circle(frame, center, radius, color, 2)

                            # declara uma fonte para usar no nome
                            font = cv2.FONT_HERSHEY_DUPLEX

                            # coordenada do nome a ser exibido
                            label_position = (x, y - 10)

                            # escreve o nome
                            cv2.putText(frame, user, label_position,
                                        cv2.FONT_HERSHEY_DUPLEX, .6, color, 2)

    # exibe a imagem
    cv2.imshow("RECONHECIMENTO FACIAL", frame)

    # aguarda alguma tecla
    key = cv2.waitKey(1)

    # para o sistema qando pressiona "q"
    if key == ord('q'):
        break
    if key == ord('Q'):
        break

# libera a camera ou vídeo
cam.release()

# destrói as janelas do opencv
cv2.destroyAllWindows()
