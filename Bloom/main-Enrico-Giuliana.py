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
THRESHOLD = 0.5015
SIGMA = 1
TIMES = 12

def brightMask(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img2[:,:, 1] = np.where(img2[:,:,1] < THRESHOLD, 0, img2[:,:,1])
    return cv2.cvtColor(img2, cv2.COLOR_HLS2BGR)

def gaussianBlurMask(mask, times = TIMES):
    sum_blurred = cv2.GaussianBlur(mask, (0,0), SIGMA)
    for i in range(1, times):
        blurred_mask = cv2.GaussianBlur(mask, (0,0), SIGMA*i*2)
        sum_blurred += blurred_mask 
    return sum_blurred

""" def gaussianBlurMask(mask, times):
    blurred_mask = cv2.GaussianBlur(mask, (0,0), SIGMA)
    sum_blurred = blurred_mask.copy()
    for i in range(1, times):
        blurred_mask = cv2.GaussianBlur(blurred_mask, (0,0), SIGMA*i*2)
        sum_blurred += blurred_mask
    return sum_blurred """

def GaussianBloomFilter(original, mask, alpha, beta, times = TIMES):
    blurred_mask = gaussianBlurMask(mask, times)
    variavel = original*alpha+blurred_mask*beta
    cv2.imshow("final", variavel)
    cv2.imshow("original", original)
    cv2.imshow("blurred_mask", blurred_mask)

def main ():
    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255

    mask = brightMask(img)
    cv2.imshow("mask", mask)

    GaussianBloomFilter(img, mask, 0.7, 0.1)

    """ cv2.imwrite ('out.png', img_out*255) """

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================



