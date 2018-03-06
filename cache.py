import os
import cv2
import numpy as np
import multiprocessing as mp
import time
def screenshot():
    os.system("adb shell /system/bin/screencap -p /mnt/sdcard/screenshot.png")
    os.system('adb pull /sdcard/screenshot.png D:\jumpjump')
def findstart(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    role=np.load('role.npy')
    h,w=role.shape
    res=cv2.matchTemplate(gray,role,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    pt=zip(*loc[::-1])
    out=[]
    for i in pt:
        out=[i[0]+int(w/2),i[1]+h]
        break
    return out
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
    for i in pointlist:
        x, y, w, h = i
        print(i)
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)
    cv2.imshow('dsf', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pointlist

def backrole(img,ob):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    role=np.load('role.npy')
    h, w = role.shape
    res = cv2.matchTemplate(gray, role, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    pt = zip(*loc[::-1])
    for i in pt:
        img[i[1]:i[1]+h,i[0]:i[0]+w]=ob
    return img

def nextobject_firststep(img,obs,startp):
    def weather_point_in_area(point, area):
        if area[0] <= point[0] and area[1] <= point[1] and area[0] + area[2] >= point[0] and area[1] + area[3] >= point[
            1]:
            return True
        return False
    pointset=[]
    for i in obs:
        if weather_point_in_area(startp, i):
            for j in obs:
                if j[1]<i[1]:
                    pointset=j
                    break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np.save('nextobject.npy',gray[pointset[1]:pointset[1]+pointset[3],pointset[0]:pointset[0]+pointset[2]])
    # cv2.imshow('sdfsdf',img[pointset[1]:pointset[1]+pointset[3],pointset[0]:pointset[0]+pointset[2]])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return pointset

def backobject(img,ob):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nextobject = np.load('nextobject.npy')
    h, w = nextobject.shape
    res = cv2.matchTemplate(gray, nextobject, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    x=int(np.mean(loc[0]))
    y=int(np.mean(loc[1]))
    # pt = zip(*loc[::-1])
    # out = []
    # for i in pt:
    img[x-3:x+h+3,y-3:y+w+3]=ob
    return img

def nextobject(img,obs,startp):
    pointset=[]
    for i in obs:
        if startp[1]>i[1]:
            pointset=i
            break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np.save('nextobject.npy',gray[pointset[1]:pointset[1]+pointset[3],pointset[0]:pointset[0]+pointset[2]])
    return pointset

n=0
while 1==1:
    screenshot()
    img = cv2.imread('screenshot.png')
    cv2.imwrite('screenshot.png', img[400:])
    n+=1
    img=cv2.imread('screenshot.png')
    ob = img[4, 504].copy()
    startp = findstart(img)
    print(startp)
    img = backrole(img, ob)
    if n!=1:
        img = backobject(img, ob)
    cv2.imwrite('screenshot.png',img)
    # os.system('python bianarzeimg.py')
    obimg=cv2.imread('img.png')
    obs=findobject(obimg)
    if n==1:
        x, y, w, h =nextobject_firststep(img,obs,startp)
        cv2.imshow('sdfsd',img[y:y+h,x:x+w])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        x, y, w, h = nextobject(img, obs, startp)
        cv2.imshow('sdfsd', img[y:y + h, x:x + w])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imshow('sf',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

