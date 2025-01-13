import keras.models
import numpy as np

class ModelConsumer:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    #Realiza la predicción de los datos que se le pasen. Se recomienda que los datos sean un
    #array de numpy para que la predicción sea más rápida. Los datos deben de ser una grabación
    #de por lo menos 10 segundos.
    def predict(self, data):
        return self.model.predict(data)

model_path = './models/ecg_model.h5'
to_predict_path = 'path/to/data.npy'

consumer = ModelConsumer(model_path)
data = np.load(to_predict_path)
predictions = consumer.predict(data)
print(f"Predictions: {predictions}")