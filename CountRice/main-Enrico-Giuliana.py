#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
# 
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
def mask_hough(img_c):
    kernel = np.ones((3, 3), np.uint8)
    img_e = img_c.copy()
    # img_e = img_e.astype (np.float32) / 255
    # for i in range(100):
    #     img_e = cv2.medianBlur(img_e,3)

    mascara = mask(img_e.copy())
    # mascara = cv2.dilate(mascara, kernel)

    img_e = img_e.reshape ((img_e.shape [0], img_e.shape [1], 1))
    img_e = img_e.astype (np.float32) / 255
    multi = mascara*img_e

    cv2.imshow('imagem original',img_c)
    cv2.imshow('imagem blur',img_e)
    cv2.imshow('imagem mascara',mascara)
    cv2.imshow('imagem multi',multi)


    sobelx = cv2.Sobel(multi,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(multi,cv2.CV_64F,0,1,ksize=3)
    img_gradient = cv2.sqrt(sobelx*sobelx + sobely*sobely)
    
    # # img_gradient = cv2.adaptiveThreshold(img_gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)
    
    
    # img_gradient = img_gradient / 360

        
    # # cv2.imshow('dx',sobelx)
    # # cv2.imshow('dy',sobely)
    # # img_gradient = np.where(img_gradient < 0.4, 0, 1.0)
    # # img_dilatate = cv2.dilate(img_gradient, kernel)
    # # img_dilatate = cv2.erode(img_gradient, kernel)
    # # for i in range(0):
    # #     img_dilatate = cv2.erode(img_dilatate, kernel)
    # # for i in range(0):
    # #     img_dilatate = cv2.dilate(img_dilatate, kernel)
    
    cv2.imshow('img_gradient',img_gradient)
    # cv2.imshow('dilatado',img_dilatate)

        
def double_mask(img):

    kernel = np.ones((3, 3), np.uint8)
    mascara = mask(img)
    cv2.imshow('mascara 1', mascara)
    # img_e = img.reshape ((img.shape [0], img.shape [1], 1))
    img_e = img.astype (np.float32) / 255
    
    sobelx = cv2.Sobel(img_e,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img_e,cv2.CV_64F,0,1,ksize=3)
    img_gradient = cv2.sqrt(sobelx*sobelx + sobely*sobely)
    img_gradient = normaliza_adap(img_gradient)
    img_gradient = cv2.sqrt(img_gradient)
    img_gradient = cv2.dilate(img_gradient,kernel)
    cv2.imshow("gradiente", img_gradient)
    
    img_teste =  np.where(img_gradient < mascara,(img_gradient),mascara)
    cv2.imshow("teste", img_teste)
    img_teste =  np.where(img_teste < 0.6 ,0.0,1.0)
    cv2.imshow("teste2", img_teste)

    final = np.where(mascara+img_teste>1.8, 0, mascara)
    cv2.imshow("final", final)

    # cv2.imshow("fim", img_teste)
    # img_teste = cv2.dilate(img_teste, kernel)
    # img_teste = cv2.erode(img_teste, kernel)
    # cv2.imshow("img-di-er", img_teste)
    # img_canny = (img*mascara)*255
    # img_canny = img_canny.astype(np.uint8)
    # img_canny = cv2.Canny(img_canny,100,200)
    # cv2.imshow("img_c",img_canny)

    # img_canny = cv2.dilate(img_canny,kernel)
    # cv2.imshow("img_di",img_canny)

    # img_canny = cv2.erode(img_canny,kernel)

    # cv2.imshow("img_er",img_canny)

    
    # img_teste =  np.where(img_teste < 0.7,0.0,img_teste)
    # cv2.imshow("teste2", img_teste)
    # img_teste =  np.where(img_teste < 0.5,0,1.0)
    # img_teste = cv2.dilate(img_teste,kernel)
    # cv2.imshow("teste2", img_teste)
    # img_e = 
    # cv2.imshow('img e ', img_e/255)
    
def limiarizacao_na_mao(img):

    img_e = img.astype (np.float32) / 255
    img_media = cv2.blur(img_e,(115,115))
    img_diff = img_e - img_media
    norm_image = normaliza(img_diff)
    # norm_image = normaliza_adap(img_diff)
    # norm_image = cv2.normalize(img_diff, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    gama = (norm_image)
    gama_uint = gama*255
    gama_uint = gama_uint.astype(np.uint8)

    cv2.imshow('img_e',img_e)
    # cv2.imshow('img_media',img_media)
    cv2.imshow('img_diff',img_diff)    
    # cv2.imshow('norm_image',norm_image)    
    cv2.imshow('gama',gama)    
    
    double_mask(gama_uint)
def normaliza(img):
    img = np.where(img<0, 0, img)
    elem_min = np.amin(img)
    elem_max = np.amax(img)
    img_norm = (img - elem_min)/(elem_max-elem_min)

    print("elementos : ",elem_min,elem_max)
    # cv2.imshow('imagem_norm', img_norm)
    return img_norm
    
def normaliza_adap(img):
    kernel = np.ones((101, 10), np.uint8)
    img_min = cv2.blur(cv2.erode(img,kernel),(51,51))
    img_max = cv2.blur(cv2.dilate(img,kernel),(51,51))
    img_norm = (img - img_min)/(img_max-img_min)
    # cv2.imshow('imagem min', img_min)
    # cv2.imshow('imagem max', img_max)
    # cv2.imshow('imagem norm', img_norm)
    return img_norm

    
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
        # cv2.imshow("gradient", img_gradient)
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
        # img_fechamento = img_fechamento.reshape ((img_fechamento.shape [0], img_fechamento.shape [1], 1))
        img_fechamento = img_fechamento.astype (np.float32) / 255

        return img_fechamento


def main ():
    for INPUT_IMAGE in INPUT_IMAGES:
        # Abre a imagem em escala de cinza.
        img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

        # mask_hough(img)
        #double_mask(img)
        limiarizacao_na_mao(img)
        # normaliza_adap(img)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        

        # É uma boa prática manter o shape com 3 valores, independente da imagem ser
        # colorida ou não. Também já convertemos para float32.
        # Mantém uma cópia colorida para desenhar a saída.
        # img_out = img.reshape ((img.shape [0], img.shape [1], 1))
        # img_out = img_out.astype (np.float32) / 255
        # img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

        
        # # Segmenta a imagem.
        # if NEGATIVO:
        #     img = 1 - img
        # img = binariza (img, THRESHOLD)
        # mask_t = mask(img)
        # cv2.imshow("mascara", mask_t)
        # cv2.imshow ('01 - binarizada', mask_t)
        # cv2.imwrite ('01 - binarizada.png', mask_t*255)

        # componentes = rotula (mask_t, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
        # n_componentes = len (componentes)
        # print ('%d componentes detectados.' % n_componentes)


        # # Mostra os objetos encontrados.
        # for c in componentes:
        #     cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,1,0))
        # cv2.imshow ('img_out', img_out)
        # cv2.imwrite (INPUT_IMAGE.replace('.','_t_e_3.'), img_erode)

        cv2.waitKey ()
        cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
