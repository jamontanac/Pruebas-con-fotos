#correrlo como python3 Basics_imutils.py
import matplotlib.pyplot as plt
import imutils
import cv2

# cargamos las fotos
bridge = cv2.imread("./Fotos/bridge.jpg")
cactus = cv2.imread("./Fotos/cactus.jpg")
logo = cv2.imread("./Fotos/pyimagesearch_logo.jpg")
workspace = cv2.imread("./Fotos/workspace.jpg")

# Intentamos primero las translaciones
#mostramos la original

plt.imshow(workspace, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title("Original")
plt.show()

# la rotamos en x 50 pixeles a la izquierda y en y 100 pixeles abajo

translated = imutils.translate(workspace, -50, 100)
plt.imshow(translated, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title("Transladada")
plt.show()
# TRansladamos ahora la imagen x =25 a la derecha y y = 75 arriba
translated = imutils.translate(workspace, 25, -75)
plt.imshow(translated, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title("Transladada 2")
plt.show()


#cv2.imshow("Original", workspace)
#cv2.waitKey(0)#para cerrar la ventana se debe presionar 0
#cv2.destroyAllWindows()


#Rotaciones
#Hacemos un loop con distintas rotaciones de la imagen
for angle in range(0, 360, 30):
    rotated = imutils.rotate(bridge, angle=angle)
    plt.imshow(rotated, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title("rotada en "+str(angle)+" grados")
    plt.show()

#Cambiando el tamaño
#Hacemos un loop para diferentes resoluciones de pixeles
for width in (400, 300, 200, 100):
    # resize the image and display it
    resized = imutils.resize(workspace, width=width)
    plt.imshow(resized,cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
    plt.title("Width=%dpx" % (width))
    plt.show()
  
#Skeletonizatrion
#encuentra la esquletonizacion de la imagen
plt.imshow(logo,cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.title("Original")
plt.show()

gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
skeleton = imutils.skeletonize(gray,size=(3,3),structuring=cv2.MORPH_RECT)
plt.imshow(skeleton,cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.title("Skeleton")
plt.show()

#Forma incorrecta de mostrar una imagen
plt.figure("Incorrecto")
plt.imshow(cactus)
#forma correcta
plt.figure("Correcto")
plt.imshow(imutils.opencv2matplotlib(cactus))
plt.show()

# Imagen de una URL
# Cargar una imagen desde la URL, convertirla al formato de openCV y muestrelo
url = "http://pyimagesearch.com/static/pyimagesearch_logo_github.png"
logo = imutils.url_to_image(url)
plt.figure("Mapeo automatico de los bordes")
plt.imshow(imutils.opencv2matplotlib(logo))
plt.show()




#Canny
#Detectar automaticamente bordes
#esta funcion detecta bordes, resulta util para lo que se quiere hacer

gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
edgeMap = imutils.auto_canny(gray)
plt.figure("Original")
plt.imshow(imutils.opencv2matplotlib(logo))
plt.show()

plt.figure("Mapeo automatico de los bordes")
plt.imshow(imutils.opencv2matplotlib(edgeMap))
plt.show()

