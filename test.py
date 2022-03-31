import cv2
import numpy as np
import os
def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    # & 输出一个 rows * cols 的矩阵（imgArray）
    #print(rows,cols)
    # & 判断imgArray[0] 是不是一个list
    rowsAvailable = isinstance(imgArray[0], list)
    # & imgArray[][] 是什么意思呢？
    # & imgArray[0][0]就是指[0,0]的那个图片（我们把图片集分为二维矩阵，第一行、第一列的那个就是第一个图片）
    # & 而shape[1]就是width，shape[0]是height，shape[2]是channel
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    # & 例如，我们可以展示一下是什么含义
    #cv2.imshow("img", imgArray[0][1])

    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                # & 判断图像与后面那个图像的形状是否一致，若一致则进行等比例放缩；否则，先resize为一致，后进行放缩
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                # & 如果是灰度图，则变成RGB图像（为了弄成一样的图像）
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        # & 设置零矩阵
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    # & 如果不是一组照片，则仅仅进行放缩 or 灰度转化为RGB
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

path = "./defect_image/shadow/"
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
#
# cv2.createTrackbar("Hue Min","TrackBars",90,179,empty)
# cv2.createTrackbar("Hue Max","TrackBars",150,179,empty)
# cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",70,255,empty)
# cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Val Max","TrackBars",255,255,empty)


for root, dirs, files in os.walk(path):
    for file in files:
        oldFile = path + file
        img = cv2.imread(oldFile)
        # img = cv2.imread(path)
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # 鼠标点击响应事件
        #鼠标点选hsv图像，保存相应位置hsv参数
        def getposHsv(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print("HSV is", imgHSV[y, x])
                filename = 'dataoilyhsv.txt'
                with open(filename, 'a') as file_object:
                    file_object.write(str(imgHSV[y,x])+"\n")

        while(True):
            imgstack=stackImages(0.4,[[img,imgHSV]])
            cv2.imshow("imageHSV",imgstack)
            cv2.setMouseCallback("imageHSV", getposHsv)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break

        # while(True):
        #     h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
        #     h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
        #     s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
        #     s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
        #     v_min = cv2.getTrackbarPos("Val Min","TrackBars")
        #     v_max = cv2.getTrackbarPos("Val Max","TrackBars")
        #     #print(h_min,h_max,s_min,s_max,v_min,v_max)
        #
        #     lower = np.array([h_min,s_min,v_min])
        #     upper = np.array([h_max,s_max,v_max])
        #     mask = cv2.inRange(imgHSV,lower,upper)
        #
        #     imgResult = cv2.bitwise_and(img,img,mask=mask)
        #
        #     imgStack = stackImages(0.2,([img,imgHSV],[mask,imgResult]))
        #     cv2.imshow("imgStack", imgStack)
        #     if cv2.waitKey(1)&0xFF==27:#ESC退出
        #         break