import cv2
import numpy as np
import multiprocessing as mp
def binaimg(pic,ob):
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            if np.sum(np.fabs(ob - pic[i, j])) < 80:
                pic[i, j] = np.array([255])
            else:
                pic[i, j] = np.array([0])
    return pic
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
img=cv2.imread('screenshot.png')
ob=img[2,250].copy()
img=backrole(img,ob)
h,w,d=img.shape
step=int(h/8)
if __name__ == "__main__":
    pool = mp.Pool(processes=4)
    re=[]
    for t in range(5):
        re.append(pool.apply_async(binaimg, (img[t* step:((t+1) * step) ],ob)))
    pool.close()
    pool.join()
    t=0
    for r in re:
        img[t * step:(t + 1) * step] =r.get()
        t+=1
cv2.imwrite("img.png", img)
