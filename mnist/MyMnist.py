import os
import struct
from array import array

import cv2
import numpy as np

import config

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


def read(path='C:\\Users\\Nadav\\Desktop\\OCR'):
    # Reads Labels
    rlbl = open(os.path.join(path, 'labels'), 'rb')
    size = struct.unpack(">I", rlbl.read(4))
    lbl = array('i', rlbl.read())
    rlbl.close()

    # Reads images
    rimg = open(os.path.join(path, 'images'), 'rb')
    amt, size = struct.unpack(">II", rimg.read(8))
    images = np.zeros((amt, size))
    for i in range(0, amt):
        images[i] = array('B', rimg.read(size))
    rimg.close()

    return np.asmatrix(images, "float32"), np.asarray(lbl, "float32")


def createdb(srcpath='C:\\Users\\Nadav\\Desktop\\OCR\\tmp', dstpath='C:\\Users\\Nadav\\Desktop\\OCR\\Test'):

    # Return values
    images = list()
    labels = list()

    # Change to working dir
    os.chdir(srcpath)

    # For each dir
    for i in os.listdir(os.getcwd()):
        # For each file
        for j in os.listdir(os.path.join(os.getcwd(), i)):
            # Reads Image and transform it
            print "Preprocessing : " + str(i) + "\\" + str(j)
            img = cv2.imread(os.path.join(os.getcwd(), i, j), cv2.IMREAD_GRAYSCALE)
            img = (255-img)
            ret, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
            img = trim(img)
            img = cv2.resize(img, (config.height,config.width))
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # Adds image and label to data
            images.append(img)
            labels.append(i)

    # Cast labels to int
    labels = np.asarray(labels, 'int')

    # Writes label data
    wlbl = open(os.path.join(dstpath, 'labels'),'wb')
    wlbl.write(struct.pack(">I",len(labels)))
    for i in labels:
        wlbl.write(i)

    wlbl.close()

    # Writes images data
    wimg = open(os.path.join(dstpath, 'images'), 'wb')
    wimg.write(struct.pack(">II", len(images), images[1].size))
    for i in images:
        wimg.write(np.asarray(i, 'B'))

    wimg.close()


def trim(img):
    # Sets min and max val
    minh, minw, maxh, maxw = [int(img.shape[0]),int(img.shape[1])],[int(img.shape[0]),int(img.shape[1])] ,[0,0],[0,0]

    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j] == 255:
                if minw[1] > j:
                    minw = [i, j]
                if maxw[1] < j:
                    maxw = [i, j]
                if minh[0] > i:
                    minh = [i, j]
                if maxh[0] < i:
                    maxh = [i, j]

    # Sets box
    box = np.asarray([[maxh],[minh],[maxw],[minw]])
    x, y, w, h = cv2.boundingRect(box)

    # Returns trim image
    img = img[x:x+w,y:y+h]
    return img

def create_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = (255-img)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = trim(img)
    img = cv2.resize(img, (config.height,config.width))
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img