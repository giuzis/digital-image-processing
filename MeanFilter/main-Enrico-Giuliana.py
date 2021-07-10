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

INPUT_IMAGE =  'a.bmp'
LARGURA_JANELA = 3
ALTURA_JANELA = 13

left = lambda n: int((n - 1)/2)
right = lambda n: int((n)/2)
limits = lambda shape,alt,lar: (left(alt), shape[0] - right(alt),left(lar), shape[1]-right(lar))
          
def filtroDaMediaIngenuo (img, altura_janela, largura_janela):
    rows,cols,channels = img.shape
    img_out = img.copy()
    L, R, T, B = limits(img.shape, altura_janela, largura_janela)
    
    for i in range(L, R):
        for j in range(T, B):
            soma = np.zeros(3)
            for k in range(i-L, i+right(altura_janela)+1):
                for l in range(j-left(largura_janela), j+right(largura_janela)+1):
                    soma += img[k,l]
            img_out[i,j] = soma/(largura_janela*altura_janela)
    
    return img_out   

def complete_lines (line_size, img) :
    rows,cols,channels = img.shape
    buffer = np.zeros((img.shape), dtype = np.float64)
    for i in range(rows):
        for j in range(line_size):
            buffer[i,left(line_size)] += (img[i,j]/line_size)
        for j in range(left(line_size)+1, cols-right(line_size)):
            buffer[i,j] = buffer[i,j-1] + (img[i,j+right(line_size)] - img[i,j-left(line_size)-1])/line_size
    return cv2.transpose(buffer)

def filtroDaMediaSeparavel(img, altura_janela, largura_janela):
    buffer = complete_lines (largura_janela,img)
    buffer = complete_lines (altura_janela, buffer)
    img_out = img.copy()
    L, R, T, B = limits(img.shape, altura_janela, largura_janela)
    img_out[L:R, T:B] = buffer[L:R, T:B] 
    return img_out



def filtroDaMediaSeparavel2(img, altura_janela, largura_janela):
    rows,cols,channels = img.shape
    buffer = np.zeros((img.shape), dtype = np.float64)
    print(rows, cols)
    for i in range(rows):
        acm = np.zeros(3)
        for j in range(largura_janela-1):
            acm += img[i,j]
        for j in range(left(largura_janela), cols-right(largura_janela)):
            acm += img[i,j+right(largura_janela)]
            buffer[i,j] = acm / largura_janela
            acm -= img[i,j-left(largura_janela)]  
    
    img_out = img.copy()

    for j in range(left(largura_janela), cols-right(largura_janela)):
        acm = np.zeros(3) 
        for i in range(altura_janela-1):
            acm += buffer[i,j]
        for i in range(left(altura_janela), R):
            acm += buffer[i+right(altura_janela),j]
            img_out[i,j] = acm / altura_janela
            acm -= buffer[i-left(altura_janela),j] 
    
    return img_out
            

def main ():
    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255

    img_ingenuo = filtroDaMediaIngenuo (img, ALTURA_JANELA, LARGURA_JANELA)
    img_separavel = filtroDaMediaSeparavel (img, ALTURA_JANELA, LARGURA_JANELA)
    
    cv2.imshow ('a - entrada', img)
    cv2.imshow ('a - ingenuo', img_ingenuo)
    cv2.imshow ('a - separavel', img_separavel)
    cv2.imwrite ('a - ingenuo.png', img_ingenuo*255)
    cv2.imwrite ('a - separavel.png', img_separavel*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
