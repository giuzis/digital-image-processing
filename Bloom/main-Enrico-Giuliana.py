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

INPUT_IMAGE =  'GT2.BMP'
# INPUT_IMAGE =  'Wind Waker GC.bmp'
THRESHOLD = 0.5015
SIGMA = 10
WINDOW = 10
TIMES = 4

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

def blurMask(mask, times = TIMES):
    sum_blurred = np.zeros(mask.shape)
    for i in range(1, times+1):
        blurred_mask = cv2.blur(mask, (WINDOW*i*2,WINDOW*i*2))
        for j in range(2):
            blurred_mask = cv2.blur(blurred_mask, (WINDOW*i*2,WINDOW*i*2))
        sum_blurred += blurred_mask 
    return sum_blurred

def GaussianBloomFilter(original, mask, alpha, beta, times = TIMES):
    blurred_mask = gaussianBlurMask(mask, times)
    blurred_img = original*alpha+blurred_mask*beta
    return blurred_img

def blurBloomFilter(original, mask, alpha, beta, times = TIMES):
    blurred_mask = blurMask(mask, times)
    blurred_img = original*alpha+blurred_mask*beta
    return blurred_img

def main ():
    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255

    mask = brightMask(img)

    gaussian_blur_img = GaussianBloomFilter(img, mask, 0.85, 0.15)
    blur_img = blurBloomFilter(img, mask, 0.85, 0.15)

    cv2.imshow ('original', img)
    cv2.imshow ('gaussian_blur_img.png', gaussian_blur_img)
    cv2.imshow ('normal_blur_img.png', blur_img)
    cv2.imwrite ('gaussian_blur_img.png', gaussian_blur_img*255)
    cv2.imwrite ('normal_blur_img.png', blur_img*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================



