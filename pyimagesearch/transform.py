# Importamos los paquetes necesarios
import numpy as np
import cv2
def order_points(pts):
    '''
    Se inicializa una lista de coordenadas que seran
    ordenadas tal que la primera entrada en la lista es
    la parte superior izquierda, la segunda entrada es la 
    parte superior derecha la tercera es la parte inferior
    derecha y  la cuarta es la inferior izquierda
    '''
    rect = np.zeros((4,2),dtype="float32")
    
    '''
    La parte superior izquierda tendra la suma mas pequeña, mientras
    que la parte inferior derecha tendra la suma mas grande
    '''
    
    s= pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]
    
    '''
    Ahora, se calcula la diferencia entre los puntos, la parte superior
    derecha tendra la diferencia mas pequena, mientras que la inferior 
    izquierda tendra la diferencia mayor
    '''
    diff = np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]
    
    #Retorna el orden de las coordenadas
    return rect

def four_point_transform(image,pts):
    '''
    Obtener un orden consistente de los puntos y 
    sacarlos de forma individual
    '''
    rect = order_points(pts)
    (tl,tr,br,bl)=rect
    '''
    Calcular el ancho de la nueva imagen a generar,
    la cual sera la distancia maxima entre la parte 
    inferior derecha y la parte inferior izquierda
    en las cordenadas de x o entre la parte superior 
    derecha a la parte superior izquierda en el eje x
    '''
    widthA = np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA),int(widthB))
    
    '''
    Calcular la altura de la nueva imagen, la cual sera la distancia 
    maxima entre la parte superior derecha y la parte inferior derecha
    en las coordenadas y o la parte superior izquierda y la parte 
    inferior izquierda tambien en las coordenadas de y
    '''
    heightA = np.sqrt((tr[0]-br[0])**2+(tr[1]-br[1])**2)
    heightB = np.sqrt((tl[0]-bl[0])**2+(tl[1]-bl[1])**2)
    maxHeight=max(int(heightA),int(heightB))
    
    '''
    Ahora que se tienen las dimensiones de la imagena crear, construimos
    el conjunto de puntos destinados a formar la imagen para obtener el
    "birds eye view" (vista desde arriba) de la imagen, de nuevo especificando
    el orden de los puntos en la parte superior izquierda, superior derecha, 
    inferior derecha e inferior izquierda.
    '''
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")
    # Calcule la matriz de transformacion de la perspectiva y apliquela
    M=cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
    
    return warped