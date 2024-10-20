import face_recognition
import cv2 as cv
import torch
import torch.nn as nn
import pyautogui
import numpy as np
import time
from ResNet import Model
import torchvision
from PIL import Image
import random
from Conv2Net import ConvNet 
from fer import FER

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pyautogui.FAILSAFE=False

def eye_aspect_ratio(eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        ear = (A + B) / (2.0 * C)
        return ear

def process(image1,image2):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Convertir imágenes a tensores
    ])
      
    derecho_image = transform(image1)
    izquierdo_image = transform(image2)

    

    return derecho_image, izquierdo_image
      


def maxAndMin(featCoords, mult=1):
    adjx = 10 / mult
    listX = []
    listY = []

    # Separar las coordenadas X e Y de las tuplas en listas separadas.
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])

    # Calcular los valores mínimos y máximos ajustados para X e Y.
    minX = min(listX) - adjx
    minY = min(listY) - adjx
    maxX = max(listX) + adjx
    maxY = max(listY) + adjx

    # Calcular el ancho y el alto de la región.
    width = maxX - minX
    height = maxY - minY

    # Asegurar que la región sea cuadrada tomando la longitud máxima.
    max_side = max(width, height)

    # Ajustar los límites para que la región sea cuadrada.
    # Centramos el cuadrado alrededor del punto central inicial.
    centerX = (minX + maxX) / 2
    centerY = (minY + maxY) / 2

    # Recalcular los límites mínimos y máximos para que sean un cuadrado.
    minX = centerX - (max_side / 2)
    maxX = centerX + (max_side / 2)
    minY = centerY - (max_side / 2)
    maxY = centerY + (max_side / 2)

    # Crear el array con los límites ajustados y multiplicarlo por `mult` para mantener la escala.
    maxminList = np.array([minX, minY, maxX, maxY])

    # Calcular las coordenadas del centro del cuadrado.
    center_coords = np.array([centerX - minX, centerY - minY])

    # Multiplicar los resultados por `mult` para escalar y convertir a enteros.
    return (maxminList * mult).astype(int), (center_coords * mult).astype(int)

def eyetrack(xshift = 70, yshift=60, frameShrink = 0.55):
    ancho_pantalla, alto_pantalla = pyautogui.size()
    print("Funcionando")
    model= ConvNet().to(device)
    model.load_state_dict(torch.load("Models/Eye100.plt",map_location=device))
    model.eval()


    webcam = cv.VideoCapture(0)
    input('Presiona Enter...')

    last_time = time.time()  
    pyautogui.moveTo(ancho_pantalla/2, alto_pantalla/2)
    EYE_CLOSED_THRESHOLD = 0.2
    detector = FER()
    while True:

        ret, frame = webcam.read()
        feats = face_recognition.face_landmarks(frame)

        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        emotion_results = detector.detect_emotions(frame)

        if emotion_results:
            emotion = emotion_results[0]['emotions']
            print(emotion['happy'])
            if emotion['happy'] > 0.5 :  
                print('Se da click')
                pyautogui.click()
        if time.time() - last_time >= 5:
            last_time = time.time()
            if len(feats) > 0:

                leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1)
                reBds, leCenter = maxAndMin(feats[0]['right_eye'], mult=1)


                right_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
                left_eye=frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]

                right_eye = cv.resize(right_eye, dsize=(128, 128))
                left_eye = cv.resize(left_eye, dsize=(128,128))

                cv.imwrite('Prueba/derecho.jpg',right_eye)
                cv.imwrite('Prueba/izquierdo.jpg',left_eye)

                image1 = Image.open('Prueba/derecho.jpg')  
                image2  = Image.open('Prueba/izquierdo.jpg')  

                left_eye_ratio = eye_aspect_ratio(image1)
                right_eye_ratio = eye_aspect_ratio(image2)

                image1,image2 = process(image1,image2)
                image1 = image1.unsqueeze(0) 
                image2 = image2.unsqueeze(0) 
                

                x=model(image1 , image2)

                probabilidades = nn.functional.softmax(x, dim=1)
                _, preds = torch.max(x, 1)

                pred_prob = probabilidades[0, preds.item()].item()

                currentMouseX, currentMouseY = pyautogui.position()
                random_number = random.randint(50, 100)
                if left_eye_ratio < EYE_CLOSED_THRESHOLD or right_eye_ratio < EYE_CLOSED_THRESHOLD:
                     print('Moviendo Abajo' + str(random_number) + ' Pixeles')
                     pyautogui.moveTo(currentMouseX , currentMouseY + random_number )
                elif preds.item()==1:
                    print('Moviendo Al Centro')
                    pyautogui.moveTo(ancho_pantalla/2,alto_pantalla/2)
                elif preds.item()==2:
                     print('Moviendo Derecha' + str(random_number) + ' Pixeles')
                     pyautogui.moveTo(currentMouseX + random_number , currentMouseY  )
                elif preds.item()==3:
                     print('Moviendo Izquierda' + str(random_number) + ' Pixeles')
                     pyautogui.moveTo(currentMouseX - random_number , currentMouseY  )
                elif preds.item()==0:
                     print('Moviendo Arriba' + str(random_number) + ' Pixeles')
                     pyautogui.moveTo(currentMouseX , currentMouseY - random_number )
                     
                     

                print(f"Predicción: {preds.item()}, Probabilidad: {pred_prob:.4f}")
        

            
            
            
try:
    eyetrack()
except Exception as e:
     print(f"Ocurrío un error : {e}")
     input("Presiona Enter para salir...")