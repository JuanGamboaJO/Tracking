import os
import cv2 as cv
import face_recognition
import pyautogui
import copy
import random
import time

import numpy as np

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


pyautogui.FAILSAFE=False
def getEye(times = 1,frameShrink = 0.3, counterStart = 0,direccion='Izquierda'):
    print('Funcionando')
    webcam = cv.VideoCapture(0)
    counter = counterStart
    counter2 =0 
    input('Presione Enter')
    while counter < counterStart+times:
        ret, frame = webcam.read()
        
        feats = face_recognition.face_landmarks(frame)
        if len(feats) > 0:
            counter+=1
            counter2 +=1
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1)
            reBds, leCenter = maxAndMin(feats[0]['right_eye'], mult=1)

            right_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            left_eye=frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]


            right_eye = cv.resize(right_eye, dsize=(128, 128))
            left_eye = cv.resize(left_eye, dsize=(128,128))

            cv.imshow('frame', left_eye)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            cv.imwrite('Ojos/' + direccion + "/" + 'Izquierdo/' +direccion  +'_'+ str(counter) + ".jpg", left_eye)
            print("Se guardo imagen del ojo izquierdo")
            cv.imwrite('Ojos/' + direccion + "/" + 'Derecho/' + direccion +'_'+ str(counter) + ".jpg", right_eye)
            print("Se guardo imagen del ojo derecho")
            if counter2==30:
                input('Presione Enter')
                counter2=0
            
        
        
getEye(times = 1000)