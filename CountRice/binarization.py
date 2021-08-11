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
kernel_completo = np.ones((3,3),np.uint8)
kernel_cruz = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
kernel_x = np.array([[1,0,1],[0,1,0],[1,0,1]],np.uint8)
img_limits = lambda i , j , s: (i >= 0 and i < s[0] and j >= 0 and j < s[1]) 


            
def canny(img, vis = False):
    mask = limiarizacao_na_mao(img)
    dilate_mask = cv2.dilate(mask,kernel_completo,iterations=2)
    img_e = np.where (dilate_mask > 0.5, (img*255).astype(np.uint8), 0 )
    suave = cv2.GaussianBlur(img_e, (7, 7), 0)
    canny1 = cv2.Canny(suave, 0, 120)
    dilate1 = canny1.copy()
    # canny2 = cv2.Canny(suave, 70, 200)
    dilate1 = cv2.dilate(canny1,kernel_cruz,iterations=3)
    dilate1 = cv2.erode(dilate1,kernel_cruz,iterations=3)
    # dilate2 = cv2.dilate(canny2,kernel_completo)
    final = np.where( dilate1 > 125, 0.0, mask)
    return final
    if vis : 
        cv2.imshow("mask",mask)
        cv2.imshow("canny1",canny1)
        # cv2.imshow("canny2",canny2)
        cv2.imshow("dilate1",dilate1)        
        # cv2.imshow("dilate2",dilate2)
        cv2.imshow("final",final)


def superborda(img, vis = False):
    grad = gradiente(img)
    mask = limiarizacao_na_mao(img)
    img_lim = np.where(mask < grad, mask, grad)
    img_lim = (img_lim)**(0.25)
    try_test = np.where(img_lim > 0.57, 0.0 , (img_lim))
    fechamento = cv2.dilate(try_test,kernel_completo)
    fechamento = cv2.erode(fechamento,kernel_completo)
    abertura = cv2.erode(try_test,kernel_completo)
    abertura = cv2.dilate(abertura,kernel_completo)
    # mediana = cv2.medianBlur((try_test*255).astype(np.uint8), 5)
    canny(try_test,True)
    if vis:
        cv2.imshow("grad", grad)
        cv2.imshow("mask", mask)
        cv2.imshow("lim", img_lim)
        cv2.imshow("try_test", try_test)
        # cv2.imshow("abertura", abertura)
        # cv2.imshow("fechamento", fechamento)
        # cv2.imshow("mediana", mediana)


        

def gradiente(img, vis = False):
    img_e = img.copy()    
    sobelx = cv2.Sobel(img_e,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img_e,cv2.CV_64F,0,1,ksize=3)
    img_gradient = cv2.sqrt(sobelx*sobelx + sobely*sobely)
    img_gradient = normaliza(img_gradient)
    if vis:
        cv2.imshow("gradiente", img_gradient)
        # cv2.imshow("gradiente_adap", img_gradient_adap)
    return img_gradient
        

#==============================================================================

def normaliza(img_or):
    img = img_or.copy()
    img = np.where(img < 0, 0, img)
    elem_min = np.amin(img)
    elem_max = np.amax(img)
    img_norm = (img - elem_min)/(elem_max-elem_min)
    print(">> ", elem_max, elem_min)
    return img_norm
    
 
def limiarizacao_na_mao(img,it=1,vis=False):
    img_e = img.copy() # cv2.blur(img.copy(),(3,3))
    img_media = cv2.blur(img_e,(115,115))
    img_diff = np.where( img_e > img_media, (img_e-img_media), 0)
    img_diff = (img_diff)**(1.2)
    img_norm = normaliza(img_diff)
    _, img_thresh = cv2.threshold((img_norm*255).astype(np.uint8) , 0, 255, cv2.THRESH_OTSU)
    img_thresh = img_thresh.astype(np.float32) / 255
    img_erode = cv2.erode(img_thresh,kernel_completo,iterations=it)
    abertura = cv2.dilate(img_erode,kernel_completo,iterations=it)
    if vis : 
        cv2.imshow('img_original',img_e)
        # cv2.imshow('img_media',img_media)
        # cv2.imshow('img_diff',img_diff)
        # cv2.imshow('img_norm',img_norm)    
        # cv2.imshow('img_thresh',img_thresh)
        # cv2.imshow('img_erode',img_erode)
        cv2.imshow('abertura',abertura)
    return abertura
    
def limiar_adap_opencv (img,vis = False):
    img_out = (img*255).astype("uint8")
    img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -25)
    img_out = img_out.astype (np.float32) / 255
    img_erode = cv2.erode(img_out,   kernel_completo)
    abertura = cv2.dilate(img_erode, kernel_completo)
    if vis:
        # cv2.imshow('img_out',img_out)
        # cv2.imshow('img_erode',img_erode)
        cv2.imshow('abertura-open',abertura)        
    return abertura


def normaliza_adap(img, k_size = 101, bw_size = 51,  vis=False):
    kernel = np.ones((k_size, k_size), np.uint8)
    img_min = cv2.blur(cv2.erode(img,kernel),(bw_size,bw_size))
    img_max = cv2.blur(cv2.dilate(img,kernel),(bw_size,bw_size))
    img_norm = (img - img_min)/(img_max-img_min)
    if vis : 
        # cv2.imshow('imagem min', img_min)
        # cv2.imshow('imagem max', img_max)
        cv2.imshow('****imagem norm', img_norm)
    
    return img_norm

def main ():
    for INPUT_IMAGE in INPUT_IMAGES:
        img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        
        img = img.astype (np.float32) / 255
        # # normalizações, gostei da primeira
        # img_n_adap = normaliza_adap(img,5,3)
        # cv2.imshow('img_n_adap0',img_n_adap)
        # img_n_adap = normaliza_adap(img,11,5)
        # cv2.imshow('img_n_adap0',img_n_adap)
        # img_n_adap = normaliza_adap(img,21,11)
        # cv2.imshow('img_n_adap1',img_n_adap)
        # img_n_adap = normaliza_adap(img,41,21)
        # cv2.imshow('img_n_adap2',img_n_adap)
        # img_n_adap = normaliza_adap(img,161,81)
        # cv2.imshow('img_n_adap3',img_n_adap)

        # # limiarização adaptativa opencv
        # limiar_adap_opencv (img,True)

        # # Limiarização adaptativa nossa
        # limiarizacao_na_mao(img,1,True)

        # Gradiente
        # gradiente(img,True)
        # superborda(img,True)
        canny(img,True)

        cv2.waitKey ()
        cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
