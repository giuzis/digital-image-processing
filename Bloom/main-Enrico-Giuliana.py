#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu, Enrico Manfron, Giuliana Martins Silva
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
from numpy.core.fromnumeric import shape
import cv2

#===============================================================================

INPUT_IMAGE =  'GT2.bmp'
THRESHOLD = 0.7

def brightMask(img):
    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            if(img[i,j,2] < THRESHOLD):
                img[i,j,2] = 0
    return img

def main ():
    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    img_out = brightMask(img)

    #start_time = timeit.default_timer ()
    
    #print ('Tempo - Filtro Media Ingenuo: %f' % (timeit.default_timer () - start_time))
    
    cv2.imwrite ('out.png', img_out)

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================
