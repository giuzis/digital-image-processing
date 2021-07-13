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

def filtroDaMediaIngenuo (img, largura_janela, altura_janela):
    rows,cols,channels = img.shape
    img_out = np.zeros([rows, cols, channels], dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            somaR = somaG = somaB = 0.0
            if(i >= np.floor(largura_janela/2) and j >= np.floor(altura_janela/2) and i <= (img.shape [0] - np.floor(largura_janela/2)) and j <= (img.shape [1] - np.floor(altura_janela/2))):
                for k in range(i-int(largura_janela/2), i+int(largura_janela/2)):
                    for l in range(j-int(altura_janela/2), j+int(altura_janela/2)):
                        somaR += img[k][l][0]
                        somaG += img[k][l][1]
                        somaB += img[k][l][2]
            img_out[i][j][0] = somaR/(largura_janela*altura_janela)
            img_out[i][j][1] = somaG/(largura_janela*altura_janela)
            img_out[i][j][2] = somaB/(largura_janela*altura_janela)
    return img_out          

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255

    img = filtroDaMediaIngenuo (img, ALTURA_JANELA, LARGURA_JANELA)
    cv2.imshow ('a - ingenuo', img)
    cv2.imwrite ('a - ingenuo.png', img*255)

    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
