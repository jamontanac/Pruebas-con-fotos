# forma de utilizar
# python transform_example.py --image Imagenes/example_01.png --cords "[(73,239),(356,117),(475,265),(187,443)]"

# importar las cosas necesarias

from pyimagesearch.transform import four_point_transform
import numpy as np
import argparse
import cv2
import matplotlib.pylab as plt

# contruir el "parse" y sus argumentos

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",help="direccion donde esta la imagen")
ap.add_argument("-c","--coords",help="lista de los puntos separados por comas")
args=vars(ap.parse_args())
'''
carguemos la imagen y tomemos las cordenadas  (lista de los (x,y))
Esto debe ser automatizado para poder ser usado de forma autonoma
'''
image=cv2.imread(args["image"])
pts = np.array(eval(args["coords"]),dtype="float32")
print(pts)
# aplicar la transformación de 4 puntos para obtener la imagen con la pperspectiva deseada
warped = four_point_transform(image,pts)
# muestre las imagenes originales y la transformación
plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
plt.imshow(warped, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
#cv2.imshow("Original",image)
#cv2.imshow("Warped",warped)
#cv2.waitkey(0)