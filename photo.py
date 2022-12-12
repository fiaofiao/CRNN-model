import os
import shutil
from PIL import Image
from turtle import pd, shape
import nrrd
import numpy as np
from sympy import Q
from torch import layer_norm

def boundingbox(img: np.ndarray, expand=1):
    (x, y) = img.shape
    cs, ce, rs, re = 0, 0, 0, 0
    for i in range(x):
        if (img[i, :]!=0).any() and rs == 0:
            rs = i
        if (img[i, :]==0).all() and rs != 0:
            re = i
            break
    for i in range(y):
        if (img[:, i]!=0).any() and cs == 0:
            cs = i
        if (img[:, i]==0).all() and cs != 0:
            ce = i
            break
    
    center = (int((rs+re)/2), int((cs+ce)/2))
    edge = int(max(ce-cs, re-rs)/2*expand)
    
    # 左上右下
    bound = (center[1]-edge, center[0]-edge, center[1]+edge, center[0]+edge)

    return bound, center

def brim(img: np.ndarray, thick=5):
    bound, center = boundingbox(img)
    brimcordis = []

    # 画🔺扩展边缘 🙂
    def _smoothing(img:np.ndarray, pixel:tuple, thin:int, direction:int):
        if thin < 1:
            return
        if direction == 0: #left-top
            img[pixel[0], pixel[1]-thin:pixel[1]]=1
            img[pixel[0]-thin:pixel[0], pixel[1]]=1
            _smoothing(img, (pixel[0]-1, pixel[1]-1), thin-2, direction)
        if direction == 1: #left-bottom
            img[pixel[0], pixel[1]-thin:pixel[1]]=1
            img[pixel[0]:pixel[0]+1, pixel[1]]=1
            _smoothing(img, (pixel[0]+1, pixel[1]-1), thin-2, direction)
        if direction == 2: #right-top
            img[pixel[0], pixel[1]:pixel[1]+thin]=1
            img[pixel[0]-thin:pixel[0], pixel[1]]=1
            _smoothing(img, (pixel[0]-1, pixel[1]+1), thin-2, direction)
        if direction == 3: #right-bottom
            img[pixel[0], pixel[1]:pixel[1]+thin]=1
            img[pixel[0]:pixel[0]+1, pixel[1]]=1
            _smoothing(img, (pixel[0]+1, pixel[1]+1), thin-2, direction)
        

    for i in range(bound[1], bound[3]):
        for j in range(bound[0], bound[2]):
            if (img[i,j]*img[i-1,j]*img[i,j-1]*img[i+1,j]*img[i,j+1] == 0) and (img[i,j] != 0):
                brimcordis.append((i,j))
    for i in range(bound[1], bound[3]):
        for j in range(bound[0], bound[2]):
            if (i,j) not in brimcordis:
                img[i,j] = 0
    for i in range(bound[1], bound[3]):
        for j in range(bound[0], bound[2]):
            if (i,j) in brimcordis:
                if (i<=center[0]) and (j<=center[1]):
                    _smoothing(img, (i,j), thick, 0)
                if (i>=center[0]) and (j<=center[1]):
                    _smoothing(img, (i,j), thick, 1)
                if (i<=center[0]) and (j>=center[1]):
                    _smoothing(img, (i,j), thick, 2)
                if (i>=center[0]) and (j>=center[1]):
                    _smoothing(img, (i,j), thick, 3)

    return img

def largestmask(mask: np.ndarray):#最大层mask
    maxroi = 0
    for i in range(mask.shape[-1]):
        slice = mask[:,:,i]
        roi = np.sum(mask[:,:,i])
        if roi > maxroi:
            maxroi = roi
            largest = slice
            reference = i
    
    return largest, reference

                
if __name__ == '__main__':
    from PIL import Image
    import pandas as pd
    root ='G:\整理图'
    Ps = []
    refers = []
    bounds = []
    volums = []
    for Pnumber in os.listdir(root):
        mask = os.path.join(root, Pnumber, sorted(os.listdir(os.path.join(root,Pnumber)))[-2])
        mask, _ = nrrd.read(mask)
        volum = np.sum(mask)
        volums.append(volum)
        Ps.append(Pnumber)

        mask, reference = largestmask(mask)
        bound, _ = boundingbox(mask)
        edge = max(bound[2]-bound[0], bound[3]-bound[1])
        for i in range(6):
            if edge >= 50:
                img = os.listdir(os.path.join(root, Pnumber, sorted(os.listdir(os.path.join(root,Pnumber)))[i]))[reference]
                img = os.path.join(root, Pnumber, sorted(os.listdir(os.path.join(root,Pnumber)))[i], img)
                img = Image.open(img)
                img = img.crop(bound)
                save_dir = os.path.join('./metasis/0', Pnumber)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if not os.path.exists(os.path.join(save_dir, str(i) + '.jpg')):
                    img.save(os.path.join(save_dir, str(i) + '.jpg'))