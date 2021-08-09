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


import matplotlib.pyplot as plt

#===============================================================================

INPUT_IMAGES =  ['60.bmp', '82.bmp', '114.bmp', '150.bmp', '205.bmp']

THRESHOLD = 0.80
ALTURA_MIN = 7
LARGURA_MIN = 7
N_PIXELS_MIN = 7

kernel_completo = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
kernel_cruz = np.array([[0,1,0],[0,1,0],[0,1,0]],np.uint8)


#===============================================================================

obj_limits = lambda pixels , alt , lar , blob: (pixels <= blob['n_pixels'] and alt <= blob['B'] - blob['T'] and lar <= blob['R'] - blob['L'])
img_limits = lambda i , j , s: (i >= 0 and i < s[0] and j >= 0 and j < s[1]) 

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

def riceMask(img, method = 1):
    if method == 1:
        img_out = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)*255).astype("uint8")
        img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -25)
        img_out = img_out.astype (np.float32) / 255
        
        img_erode = cv2.erode(img_out,   kernel_completo)
        abertura = cv2.dilate(img_erode, kernel_cruz)
        
        # cv2.imshow("img_out",img_out )        
        # cv2.imshow("img_erode",img_erode )
        # cv2.imshow("abertura", abertura )        
        
        return abertura
    
    else:
        return img   
def countRice (mask , original_img = None):
    if (len(mask.shape) == 2):
        mask = mask.reshape ((mask.shape [0], mask.shape [1], 1)) 

    componentes = rotula (mask, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)

    pixels_por_componente = [c['n_pixels'] for c in componentes]
    
    y, x, _ = plt.hist(pixels_por_componente,len(componentes))
    
    n_pixels_rice = (x[np.where(y == y.max())][0])

    # plt.show()

    #n_pixels_rice = n_pixels_rice #np.where(componentes_por_pixel == np.max(componentes_por_pixel))[0][0]

    media = n_pixels_rice# n_pixels_rice #np.median([c['n_pixels'] for c in componentes])
    media = np.median(pixels_por_componente) 
    # print(">>", media)
    std = np.std(pixels_por_componente)
    # print(">>", std)
    media -= (std/34)
    print ("media calculada de pixels por componente: ", media) 
    
    total = 0
    for c in componentes:
        n_rices = np.around((c['n_pixels'] / media))
        n_rices += 1 if (n_rices < 1) else 0
        total += n_rices
        c['n_rices'] = str(int(n_rices))

    if type(original_img) is np.ndarray:
        img_out = cv2.cvtColor (original_img, cv2.COLOR_BGR2GRAY)
        img_out = cv2.cvtColor (img_out, cv2.COLOR_GRAY2BGR)

        for c in componentes:
            cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,1,0))
            cv2.putText(img_out, #numpy array on which text is written
                        c['n_rices'], #text
                        (int((c['R']+c['L'])/2)-5 ,int((c['T']+c['B'])/2)+5), #position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, #font family
                        0.5, #font size
                        (0, 255, 0), #font color
                        1) #font stroke
        cv2.imshow ('result', img_out)
        # cv2.imwrite (INPUT_IMAGE.replace('.','_t_e_3.'), img_erode)
    
    return total

def main ():
    for INPUT_IMAGE in INPUT_IMAGES:
        img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)            # Abre a imagem em escala de cinza.

        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        
        img = img.astype (np.float32) / 255
        
        rice_mask = riceMask(img)

        n_components = countRice(rice_mask,img)

        print ('%d de %d componentes detectadas.' % (n_components, int(INPUT_IMAGE.split('.')[0])))

        cv2.waitKey ()
        cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
