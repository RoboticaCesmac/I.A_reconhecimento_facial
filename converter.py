import tensorflow as tf #importa o tensorflow

model = tf.keras.models.load_model('faces.h5') #carrega o modelo do tensorflow

converter = tf.lite.TFLiteConverter.from_keras_model(model) #carrega o modelo para o tensorflow lite

tflite_model = converter.convert() #converte para o tensorflow lite

open("faces.tflite", "wb").write(tflite_model) #salva o modelo tensorflow lite