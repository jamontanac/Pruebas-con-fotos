# la forma de usar es
#python3 scan.py --image Imagenes/page.jpg


# importar los paquetes que se necesitan
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pylab as plt

#Construir la parte de "parse" para correr el codigo

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path a la imagen que se quiere escanear")
args=vars(ap.parse_args())

# Cargar la imagen y calcular la porcion de la altura que se va a modificar
# clonar la imagen y reajustar el tamaño
num_pixels=700
image=cv2.imread(args["image"])
ratio = image.shape[0]/num_pixels
#Solo para mejorar el rendimiento del procesamiento,
#se pone la imagen escaneada a un tamaño de 500 pixeles
original = image.copy()
image=imutils.resize(image,height=num_pixels)
'''
plt.figure("original")
plt.imshow(imutils.opencv2matplotlib(image))
plt.show()
'''

#Convertir la imagen a una escala de grices, borrosearla,
# encontrar los bordes enb la imagen
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,75,200)
print("Paso 1 detectar bordes")
'''
plt.figure("edged")
plt.imshow(imutils.opencv2matplotlib(edged))
plt.show()
'''
plt.subplot(121)
plt.imshow(imutils.opencv2matplotlib(image))
plt.xticks([]),plt.yticks([])
plt.title("Original image")
plt.subplot(122)
plt.imshow(imutils.opencv2matplotlib(edged),cmap="gray")
plt.title("Edged  image")
plt.xticks([]),plt.yticks([])
plt.show()

# encontrar los contornos en la imagen "edged" manteniendo solo aquellos bordes más grandes

cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea,reverse = True)[:5]
#ctns contiene los contornos ordenados de menor a mayor
for c in cnts:
	#se aproxima el contorno
	peri = cv2.arcLength(c,True)
	approx = cv2.approxPolyDP(c,0.02*peri,True)
	# si nuestro contorno aproximado tiene cuatro
	# puntos, entonces se podria decir que tenemos
	# la imagen formada.
	if len(approx)==4:
		screenCtn = approx
		break
print("Paso 2 encontrar el contorno del documento")
cv2.drawContours(image,[screenCtn],-1,(255,0,50),2)
plt.imshow(imutils.opencv2matplotlib(image))
plt.xticks([]),plt.yticks([])
plt.title("Original image")
plt.show()

# Apliquemos la transformacion de cuatro puntos
warped = four_point_transform(original,screenCtn.reshape(4,2)*ratio)

#Convertir las imagenes warped a una esca de grices, luego imponer
#un umbral para dar el efecto blanco y negro del papel

warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T = threshold_local(warped,35,offset=10,method = "gaussian")
warped = (warped > T).astype("uint8")*255

print("Paso 3: Aplicar el cambio de perspectiva")

'''
plt.imshow(imutils.opencv2matplotlib(original))
plt.xticks([]),plt.yticks([])
plt.title("Original image")
plt.show()

plt.imshow(imutils.opencv2matplotlib(warped))
plt.xticks([]),plt.yticks([])
plt.title("Escaneada")
plt.show()
'''
plt.imshow(imutils.opencv2matplotlib(imutils.resize(original,height = 800)))
plt.xticks([]),plt.yticks([])
plt.title("Original imagen con ampliacion")
plt.show()

plt.imshow(imutils.opencv2matplotlib(imutils.resize(warped,height =800)))
plt.xticks([]),plt.yticks([])
plt.title("Escaneada con ampliacion")
plt.show()
