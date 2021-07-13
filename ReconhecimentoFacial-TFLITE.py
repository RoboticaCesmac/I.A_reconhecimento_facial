# É NECESSÁRIO UTILIZAR O PYTHON 3.6.0 NESSE CÓDIGO

import numpy as np
from PIL import Image
from mtcnn_tflite.MTCNN import MTCNN
from tensorflow.keras.models import load_model
import cv2
from sklearn.preprocessing import Normalizer
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

#credenciais firebase
cred = credentials.Certificate("firebase.json")

#inicializa firebase
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://reconhecimento-facial-aa406-default-rtdb.firebaseio.com/'
})

#referência database
ref = db.reference()


# detector de faces
detector = MTCNN()

# extrator de embeddings
facenet = load_model("facenet_keras.h5")

# modelo de reconhecimento de faces
model = load_model("faces.h5")

# detector de máscara
modelMask = load_model("detector_mascara.h5")

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
def get_embeddig(facenet, face_pixels):

    # normalização
    face_pixels = face_pixels.astype("float32")

    # normalização
    mean, std = face_pixels.mean(), face_pixels.std()

    # normalização
    face_pixels = (face_pixels - mean) / std

    # expnsão dee dimensões para matriz
    samples = np.expand_dims(face_pixels, axis=0)

    # predict para extrair embedding
    yhat = facenet.predict(samples)

    # retorna embedding
    return yhat[0]


# #configurações para uso sem webcam
# def rescale(im, ratio):
#     size = [int(x*ratio) for x in im.size]
#     im = im.resize(size, Image.ANTIALIAS)
#     return im

# frame_width = int(cam.get(3))
# frame_height = int(cam.get(4))

# size = (frame_width, frame_height)
# video = cv2.VideoWriter('filename.avi',
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          60, size)
# #

# loop infinito
while True:

    # ler imagem
    ret, frame = cam.read()

    # detectar faces na imagem
    faces = detector.detect_faces(frame)

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

                    # verifica se está usando máscara
                    result = modelMask.predict([[roi]])

                    # extrai o resultado
                    result = result[0]

                    # se estiver sem máscara (probabilidade de 98% ou mais)
                    if result[0] >= 0.98:

                        # normalização
                        face = face.astype("float32")/255

                        # extrai embedding
                        embedding = get_embeddig(facenet, face)

                        # transforma em uma matriz (tensor)
                        tensor = np.expand_dims(embedding, axis=0)

                        # normalização l2
                        norm = Normalizer(norm="l2")

                        # executa normalização
                        tensor = norm.transform(tensor)

                        # prediz qual é a face detectada
                        classe = model.predict_classes(tensor)[0]
                        print(classe)

                        # probabilidade de ser a face detectada
                        prob = model.predict_proba(tensor)

                        # probabilidade em porcentagem
                        prob = prob[0][classe] * 100

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

    # #salva frames em um vídeo
    # video.write(frame)

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
