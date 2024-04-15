from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Definir una lista de colores para asignar a cada número de forma
lista_colores = [(255, 255, 255),  # Blanco
                 (255, 0, 0),      # Azul
                 (0, 0, 255),      # Rojo
                 (0, 255, 0),      # Verde
                 (255, 255, 0),    # Cyan
                 (255, 0, 255)]    # Magenta

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convertir a escala de grises Importante
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detección de rostros
            faces = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Contador de rostros
            num_faces = len(faces)
            
            for (x, y, w, h) in faces:
                # Dibujar un rectángulo alrededor de cada rostro detectado
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Mostrar el número total de rostros detectados
            cv2.putText(frame, f"Rostros detectados: {num_faces}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Convertir JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
