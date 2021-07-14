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
LARGURA_JANELA = 11
ALTURA_JANELA = 11

left = lambda n: int((n - 1)/2)
right = lambda n: int((n)/2)
limits = lambda shape,alt,lar: (left(alt), shape[0] - right(alt),left(lar), shape[1]-right(lar))
          
def filtroDaMediaIngenuo (img, altura_janela, largura_janela):
    rows,cols,channels = img.shape
    img_out = img.copy()
    # img_out[:,:] = (0,1,1)
    T, B, L, R = limits(img.shape, altura_janela, largura_janela)
    
    for i in range(T, B):
        for j in range(L, R):
            soma = np.zeros(3)
            for k in range(i-T, i+right(altura_janela)+1):
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
    img_out = img.copy()
    # img_out[:,:] = (0,1,1)
    T, B, L, R = limits(img.shape, altura_janela, largura_janela)
    img_out[T:B,L:R]= complete_lines (altura_janela, buffer)[T:B,L:R]
         
    return img_out



def filtroDaMediaSeparavel2(img, altura_janela, largura_janela):
    rows,cols,channels = img.shape
    buffer = np.zeros((img.shape), dtype = np.float64)
    for i in range(rows):
        for j in range(largura_janela):
            buffer[i,left(largura_janela)] += (img[i,j]/largura_janela)
        for j in range(left(largura_janela)+1, cols-right(largura_janela)):
            buffer[i,j] = buffer[i,j-1] + (img[i,j+right(largura_janela)] - img[i,j-left(largura_janela)-1])/largura_janela
    
    img_out = img.copy()
    # img_out[:,:] = (0,1,1)
    T, B, L, R = limits(img.shape, altura_janela, largura_janela)
    img_out[T:B,L:R] = (0,0,0)
    
    for j in range(L,R):
        for i in range(altura_janela):
            img_out[left(altura_janela),j] += (buffer[i,j]/altura_janela)
        for i in range(left(altura_janela)+1, rows-right(altura_janela)):
            img_out[i,j] = img_out[i-1,j] + (buffer[i+right(altura_janela),j] - buffer[i-left(altura_janela)-1,j])/altura_janela

    return img_out

def integral (img):
    rows,cols,channels = img.shape
    integral = img.copy()
    for i in range(rows):
        for j in range(cols-1):
            integral [i,j+1] += integral[i,j]
    for j in range(cols):
        for i in range(rows-1):
            integral [i+1,j] += integral[i,j]
    return integral

def filtroDaMediaIntegral(img, altura_janela, largura_janela):
    rows,cols,channels = img.shape
    img_integral = np.zeros((rows + 1, cols + 1, channels), dtype = np.float64)
    img_integral[1:,1:] = integral(img)
    img_out = img.copy()
    # img_out[:,:] = (0,1,1)
    T, B, L, R = limits(img.shape, altura_janela, largura_janela)
    for i in range(T,B):
        for j in range(L,R):
            img_out[i,j] = ((img_integral[i + right(altura_janela) + 1, j + right(largura_janela) + 1] 
                            -img_integral[i - left(altura_janela), j + right(largura_janela) + 1]   
                            -img_integral[i + right(altura_janela) + 1, j - left(largura_janela)]  
                            +img_integral[i - left(altura_janela), j - left(largura_janela)] ) 
                            /(altura_janela*largura_janela))
    return img_out


def main ():
    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255

    start_time = timeit.default_timer ()
    img_ingenuo = filtroDaMediaIngenuo (img, ALTURA_JANELA, LARGURA_JANELA)
    print ('Tempo - Filtro Media Ingenuo: %f' % (timeit.default_timer () - start_time))
    start_time = timeit.default_timer ()
    img_separavel = filtroDaMediaSeparavel (img, ALTURA_JANELA, LARGURA_JANELA)
    print ('Tempo - Filtro Media Separavel 1: %f' % (timeit.default_timer () - start_time))
    start_time = timeit.default_timer ()
    img_separavel_2 = filtroDaMediaSeparavel2 (img, ALTURA_JANELA, LARGURA_JANELA)
    print ('Tempo - Filtro Media Separavel 2: %f' % (timeit.default_timer () - start_time))
    start_time = timeit.default_timer ()
    img_integral = filtroDaMediaIntegral(img, ALTURA_JANELA, LARGURA_JANELA)
    print ('Tempo - Filtro Media Integral: %f' % (timeit.default_timer () - start_time))


    

    cv2.imshow ('a - entrada', img)
    cv2.imshow ('a - ingenuo', img_ingenuo)
    cv2.imshow ('a - separavel', img_separavel)
    cv2.imshow ('a - separavel_2', img_separavel_2)
    cv2.imshow ('a - integral', img_integral)
    

    cv2.imwrite ('a - ingenuo.png', img_ingenuo*255)
    cv2.imwrite ('a - separavel.png', img_separavel*255)
    cv2.imwrite ('a - separavel_2.png', img_separavel_2*255)
    cv2.imwrite ('a - integral.png', img_integral*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
