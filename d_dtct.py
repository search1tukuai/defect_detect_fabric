# -*- coding = utf-8 -*-
# @Time : 2022/03/29 11:46
# @Author : yuseph
# @Software : PyCharm
import os
import cv2
import numpy as np
import math

# 变量
srcPath = "./defect_image/shadow/"  # 原始路径
dstPath = "C:/Users/98137/Desktop/res/"  # 存储路径
index = 1
type = ["oily","shadow"]
def fill_lty(img, num):
    '''
    img:输入,二值化图像
    num:按面积排序需要保留的连通域数量，例如面积最大的前10个，剩下的连通域被填充
    '''
    lty = img.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lty, connectivity=8)
    id = stats[np.lexsort(-stats.T)]
    for istat in id[num:-1]:
        if istat[3] > istat[4]:
            r = istat[3]
        else:
            r = istat[4]
        cv2.rectangle(lty, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), (0, 0, 0), thickness=-1)
    return lty

def getContours(img,res,text):
    '''
    :param img: input
    :param res: output
    :param text: the text you want to add
    :return: none
    '''
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 220000>area > 100:
            # cv2.drawContours(res, cnt, -1, (255, 0, 0), 3)
            peri =cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(res,text,
                            (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                            (0,0,0),2)

def oilydtct(img):
    oily = img.copy()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min, h_max, s_min, s_max, v_min, v_max = 48, 150, 0, 255, 0, 255
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    m_ = cv2.bitwise_not(mask)
    retval1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))  # 开运算的核
    retval2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(9, 9))  # 闭运算的核
    open = cv2.morphologyEx(m_, cv2.MORPH_OPEN, retval1)  # 先开运算
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, retval2)  # 再闭运算
    getContours(close, oily, type[0])
    return oily

def shadowdtct(img):
    img = cv2.resize(img, (612, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 133, 20)  # 阈值分割
    # kern = cv2.getGaborKernel(ksize=(51,51), sigma=12, theta=0.25*math.pi, lambd=25, gamma=2, psi=0)
    # fimg = cv2.filter2D(gray, cv2.CV_8UC3, kern);
    # ret,th=cv2.threshold(gray,135,255,cv2.THRESH_BINARY_INV)
    # can = cv2.Canny(th,150,300)
    # lty = fill_lty(th,200)
    retval1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3))  # 开运算的核
    retval2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(27, 27))  # 闭运算的核
    open = cv2.morphologyEx(th, cv2.MORPH_OPEN, retval1)  # 先开运算
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, retval2)  # 再闭运算
    lty = fill_lty(close, 6)
    getContours(lty, img, type[1])
    return img

for root, dirs, files in os.walk(srcPath):
    for file in files:
        oldFile = srcPath + file
        img = cv2.imread(oldFile)
        # res1=oilydtct(img)
        res2=shadowdtct(img)
        newName = dstPath + file + "_"+type[1] + "_" + str(index) + ".png"
        cv2.imwrite(newName, res2)
        index += 1