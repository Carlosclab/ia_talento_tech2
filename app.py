from flask import Flask, request, render_template
import pandas as pd
import requests
import joblib
from datetime import datetime
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classification_model_path = os.path.join(BASE_DIR, 'models', 'classification_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')


classification_model = joblib.load(classification_model_path)
scaler = joblib.load(scaler_path)
datos=[]

def obtener_datos(ciudad):
    URL = 'https://api.weatherapi.com/v1/current.json'
    HEADERS={'Content-Type': 'application/json', 'Accept': 'application/json'}
    PARAMETERS = {
        'key':'de3de7ff15174e0195832842241112',
        'q':ciudad,
        'aqi':'no'
    }
    response = requests.get(url=URL, params=PARAMETERS,headers=HEADERS)
    datos_api = response.json()
    return [datos_api['current']['temp_f'],datos_api['current']['pressure_in'],datos_api['current']['humidity'],datos_api['current']['wind_degree'],datos_api['current']['wind_kph'],datetime.now().hour]

@app.route('/', methods=['GET', 'POST'])
def index():
    
    prediction = None
    classification_result = None
    datos=[]
    bg_color = "#ffffff"  # color por defecto (blanco)

    if request.method == 'POST':
        input_value = request.form.get('input_value', None)
        try:
            datos = obtener_datos(input_value)
            #feature names for datos
            data = pd.DataFrame([datos],columns= ['Temperature','Pressure','Humidity','WindDirection(Degrees)','Speed','hour'])
            print(data)
            X = scaler.transform(data)
            

            # Predicción de clasificación
            class_pred = classification_model.predict(X)[0]
            print(class_pred)
            classification_result =class_pred
            
            # Cambiar el color de fondo según el resultado de clasificación
            if classification_result == "Alto":
                bg_color = "#ff333c"  # Fondo rojizo
            elif classification_result == "Medio":
                bg_color = "#fff633"
            elif classification_result == "Bajo":
                bg_color = "#33ff96"
            else:
                bg_color = "#33ffe3"  # Fondo verdoso

        except (ValueError, TypeError):
        #     print(ValueError)
            prediction = "Entrada inválida"
            classification_result = "No se pudo clasificar"
            bg_color = "#ffffff"

    return render_template('index.html', 
                           prediction=prediction, 
                           classification_result=classification_result,
                           bg_color=bg_color,info=datos)

if __name__ == "__main__":
    app.run()
