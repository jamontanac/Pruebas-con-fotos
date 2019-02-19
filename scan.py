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

image=cv2.imread(args["image"])
ratio = image.shape[0]/500
#Solo para mejorar el rendimiento del procesamiento, 
#se pone la imagen escaneada a un tamaño de 500 pixeles
original = image.copy()
image=imutils.resize(image,height=500)

plt.figure("original")
plt.imshow(imutils.opencv2matplotlib(image))
plt.show()


#Convertir la imagen a una escala de grices, borrosearla,
# encontrar los bordes enb la imagen
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,75,200)

plt.figure("edged")
plt.imshow(imutils.opencv2matplotlib(edged))
plt.show()

