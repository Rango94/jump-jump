import os
import cv2
import numpy as np
import math
import multiprocessing as mp
import time
def screenshot():
    os.system("adb shell /system/bin/screencap -p /mnt/sdcard/screenshot.png")
    os.system('adb pull /sdcard/screenshot.png D:\jumpjump')
    img = cv2.imread('screenshot.png')
    img=cv2.resize(img,(540,960))
    cv2.imwrite('screenshot.png',img)

def findstart(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    role=np.load('role.npy')
    h,w=role.shape
    role=cv2.resize(role,(int(w/2),int(h/2)))
    h, w = role.shape
    res=cv2.matchTemplate(gray,role,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    out = []
    while out==[]:
        try:
            loc = np.where( res >= threshold)
            pt=zip(*loc[::-1])
            for i in pt:
                out=[i[1]+h-5,i[0]+int(w/2)]
                break
        except:
            threshold -= 0.05
    return out
import random
def pushpush(t):
    x1=random.randint(400,1000)
    x2=random.randint(400,1000)
    y1 = random.randint(400, 1000)
    y2 = random.randint(400, 1000)
    os.system('adb shell input swipe '+str(x1)+' '+str(x2)+' '+str(y1)+' '+str(y2)+' '+str(t))

def backrole(img,ob):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    role=np.load('role.npy')
    h, w = role.shape
    role = cv2.resize(role, (int(w / 2), int(h / 2)))
    h, w = role.shape
    res = cv2.matchTemplate(gray, role, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    mark=1
    while mark==1:
        try:
            loc = np.where(res >= threshold)
            pt = zip(*loc[::-1])
            for i in pt:
                img[i[1]:i[1]+h,i[0]:i[0]+w]=ob
                mark=0
        except:
            threshold-=0.05
    return img

def findobject(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    hs, ws = binary.shape
    for i in range(hs):
        for j in range(ws):
            if i == 0 or i == hs - 1 or j == 0 or j == ws - 1:
                binary[i][j] = 255
    _,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pointlist=[]
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if (w < ws or h < hs) and w>10 and h>10:
            pointlist.append([x, y, w, h])
    i = 0
    lenth = len(pointlist)
    while i < lenth - 1:
        j = 0
        while j < lenth:
            if j == i and j != lenth - 1:
                j += 1
            if pointlist[i][0] >= pointlist[j][0] and pointlist[i][1] >= pointlist[j][1] and pointlist[i][0] + \
                    pointlist[i][2] <= pointlist[j][2] + pointlist[j][0] and pointlist[i][3] <= pointlist[j][3] - (
                        pointlist[i][1] - pointlist[j][1]):
                del pointlist[i]
                lenth = len(pointlist)
                break
            else:
                j += 1
        if j == lenth:
            i += 1
    high=1000
    pointset=0
    for i in pointlist:
        x, y, w, h = i
        if y<high:
            high=y
            pointset=i
    x, y, w, h =pointset
    pic=binary[y:y+h,x:x+w]
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            if pic[i,j]==0 and (j + x < startp[1] - 3 or j + x > startp[1] + 3):
                for k in range(len(pic)):
                    for l in range(len(pic[k])-1,-1,-1):
                        if pic[k, l] == 0 and (l + x < startp[1] - 3 or l + x > startp[1] + 3):
                            j=(j+l)/2
                            if j+x<startp[1]:
                                y_=0.61*(j+x-startp[1])+startp[0]
                            else:
                                y_=-0.61*(j+x-startp[1])+startp[0]
                            return [int((y_+y+i)/2),int(j+x)]
    return [0,0]

def findendp(pic,ob):
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            if np.sum(np.fabs(ob - pic[i, j])) > 80:
                for k in range(len(pic[i])-1,-1,-1):
                    if np.sum(np.fabs(ob - pic[i, k])) > 80:
                        j=(j+k)/2
                        if j  < startp[1]:
                            y_ = 0.61 * (j- startp[1]) + startp[0]
                        else:
                            y_ = -0.61 * (j- startp[1]) + startp[0]
                        return [int((y_  + i) / 2), int(j )]
    return [0,0]

while 1==1:
    screenshot()
    img = cv2.imread('screenshot.png')
    cv2.imwrite('screenshot.png', img[200:])
    img=cv2.imread('screenshot.png')
    ob = img[2, 250].copy()
    startp = findstart(img)
    # os.system('python bianarzeimg.py')
    # obimg=cv2.imread('img.png')
    # endp=findobject(obimg)
    img=backrole(img,ob)
    endp=findendp(img,ob)
    print(startp,endp,np.fabs(np.array(startp)-np.array(endp)))
    dis=np.sum(np.square(np.fabs(np.array(startp)-np.array(endp))))
    print(dis)
    y=2.241*math.pow(dis,0.5138)
    print(y)
    pushpush(int(y))
    img[endp[0]-5:endp[0]+5,endp[1]-5:endp[1]+5]=np.array([0,0,255])
    cv2.imshow('sf',img)
    cv2.waitKey(1)
    time.sleep(y/1000+1+0.5*random.random())
#     [321, 187] [236, 135] [ 85.  52.]

