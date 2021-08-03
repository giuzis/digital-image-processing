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

INPUT_IMAGES =  ['60.bmp', '82.bmp', '114.bmp', '150.bmp', '205.bmp']

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.80
ALTURA_MIN = 7
LARGURA_MIN = 7
N_PIXELS_MIN = 7

#===============================================================================

def binariza (img, threshold):
    return np.where(img < threshold, 0, 1.0)
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

#-------------------------------------------------------------------------------
def obj_limits(pixels , alt , lar , bloob):
    return  pixels <= bloob['n_pixels'] and alt <= bloob['B'] - bloob['T'] and lar <= bloob['R'] - bloob['L']

def img_limits(i , j , s):
    return i >= 0 and i < s[0] and j >= 0 and j < s[1] 

def flood (data, img, i, j):
    img[i][j][0] = 1.1
    data['n_pixels'] += 1
    for di, dj, dr, df, dv in {(0,1,'R',max,j),(0,-1,'L', min,j),(1,0,'B', max, i),(-1,0,'T', min, i)}:
        if (img_limits(i+di,j+dj,img.shape) and img[i+di][j+dj] == 1): 
            data[dr] = df(data[dr],dv + 2*(di + dj))
            flood(data, img, i + di, j + dj)
    

def rotula (img, largura_min, altura_min, n_pixels_min):
    labels = []
    for i, j, _ in np.argwhere(img > THRESHOLD): 
        if img[i][j][0] == 1:
            bloob = {'label': len(labels)+1,'n_pixels':0,'T':i,'L':j,'B':i,'R':j}  
            flood(bloob,img,i,j)
            if (obj_limits(n_pixels_min, altura_min, largura_min, bloob)):
                labels.append(bloob)
        else:
            img[i][j][0] = 1.1
    return labels

    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.

#===============================================================================

def main ():
    for INPUT_IMAGE in INPUT_IMAGES:
        # Abre a imagem em escala de cinza.
        img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        # cv2.imshow ('original', img)
        # img = cv2.medianBlur(img,3)

        img_out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -25)
        cv2.imshow ('THRESHOLD', img_out)
        # laplacian = cv2.Laplacian(img,cv2.CV_64F)
        img = img.reshape ((img.shape [0], img.shape [1], 1))
        img = img.astype (np.float32) / 255
        
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        integrate = cv2.sqrt(sobelx*sobelx + sobely*sobely)*255

        cv2.imshow('dx',sobelx)
        cv2.imshow('dy',sobely)
        cv2.imshow('integrate',integrate)
        cv2.imshow('borderless', (img_out - integrate))

        # kernel = np.ones((3, 3), np.uint8)
        # img_erode = cv2.erode(img_out, kernel)
        # img_erode = cv2.erode(img_erode, kernel)
        # img_erode = cv2.erode(img_erode, kernel)

        # img = cv2.GaussianBlur(img, (0,0), 1.1)
        # É uma boa prática manter o shape com 3 valores, independente da imagem ser
        # colorida ou não. Também já convertemos para float32.
        # Mantém uma cópia colorida para desenhar a saída.
        # img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)
        # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        # # Segmenta a imagem.
        # if NEGATIVO:
        #     img = 1 - img
        # img = binariza (img, THRESHOLD)
        # cv2.imshow ('01 - binarizada', img)
        # cv2.imwrite ('01 - binarizada.png', img*255)

        # start_time = timeit.default_timer ()
        # componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
        # n_componentes = len (componentes)
        # print ('Tempo: %f' % (timeit.default_timer () - start_time))
        # print ('%d componentes detectados.' % n_componentes)

        # # Mostra os objetos encontrados.
        # for c in componentes:
        #     cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

        # cv2.imshow ('erudida', img_erode)
        
        # cv2.imwrite (INPUT_IMAGE.replace('.','_t_e_3.'), img_erode)

        cv2.waitKey ()
        cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
