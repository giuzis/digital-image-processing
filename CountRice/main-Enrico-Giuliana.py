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
def obj_limits(pixels , alt , lar , blob):
    return  pixels <= blob['n_pixels'] and alt <= blob['B'] - blob['T'] and lar <= blob['R'] - blob['L']

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
            blob = {'label': len(labels)+1,'n_pixels':0,'T':i,'L':j,'B':i,'R':j}  
            flood(blob,img,i,j)
            if (obj_limits(n_pixels_min, altura_min, largura_min, blob)):
                labels.append(blob)
        else:
            img[i][j][0] = 1.1
    return labels

#===============================================================================
def mask (img):
        # cv2.imshow ('original', img)
        # img = cv2.medianBlur(img,3)

        # img_out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -25)
        # cv2.imshow ('THRESHOLD', img_out)

        # img = img.astype (np.float32) / 255
        
        # sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        # sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        # img_gradient = cv2.sqrt(sobelx*sobelx + sobely*sobely)
        # img_gradient = img_gradient / 1.414

        # cv2.imshow('dx',sobelx)
        # cv2.imshow('dy',sobely)
        # cv2.imshow('img_gradient',img_gradient)
        
        # kernel = np.ones((3, 3), np.uint8)

        # img_erode = cv2.erode(img_out, kernel)
        # img_dilatate = cv2.dilate(img_out, kernel)

        # img_fechamento = cv2.dilate(img_erode, kernel)
        # img_nao_fechamento = cv2.erode( img_dilatate,kernel)
        # cv2.imshow('borderless', (img_out - img_gradient))
        # cv2.imshow('fechamento',img_fechamento)
        # cv2.imshow('nao_fechamento',img_nao_fechamento)
        
        # img_out = img_out.reshape ((img_out.shape [0], img_out.shape [1], 1))
        # img_out = img_out.astype (np.float32) / 255
        # print(img_out.shape)
        # print(img_gradient.shape)
        # cv2.imshow('multi',img_out - img_gradient*5) 
        
        img_out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -25)
        kernel = np.ones((3, 3), np.uint8)
        img_erode = cv2.erode(img_out, kernel)
        img_fechamento = cv2.dilate(img_erode, kernel)
        img_fechamento = img_fechamento.reshape ((img_fechamento.shape [0], img_fechamento.shape [1], 1))
        img_fechamento = img_fechamento.astype (np.float32) / 255

        return img_fechamento


def main ():
    for INPUT_IMAGE in INPUT_IMAGES:
        # Abre a imagem em escala de cinza.
        img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        

        # É uma boa prática manter o shape com 3 valores, independente da imagem ser
        # colorida ou não. Também já convertemos para float32.
        # Mantém uma cópia colorida para desenhar a saída.
        img_out = img.reshape ((img.shape [0], img.shape [1], 1))
        img_out = img_out.astype (np.float32) / 255
        img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

        
        # # Segmenta a imagem.
        # if NEGATIVO:
        #     img = 1 - img
        # img = binariza (img, THRESHOLD)
        img = mask(img)

        # cv2.imshow ('01 - binarizada', img)
        # cv2.imwrite ('01 - binarizada.png', img*255)

        componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
        n_componentes = len (componentes)
        print ('%d componentes detectados.' % n_componentes)


        # # Mostra os objetos encontrados.
        for c in componentes:
            cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,1,0))

        cv2.imshow ('img_out', img_out)
        
        # cv2.imwrite (INPUT_IMAGE.replace('.','_t_e_3.'), img_erode)

        cv2.waitKey ()
        cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
