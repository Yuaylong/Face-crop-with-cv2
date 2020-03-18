import cv2
import os
import time

PathI = './Input Path/' #-->Input path
PathS = './Save Path/'  #-->Save path
n = 0                   #-->Define numbers in image names
ncnt = 0                #-->Count image

# Load the cascade (https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

start = time.time() #-->Start timer
for Finame in os.listdir(PathI):
    total = len(os.listdir(PathI))
    percent = ((total-ncnt)/total)*100
    
    print(f'----------total: {percent:0.2f} %----------')
    print('Process input')
    
    img = cv2.imread(PathI + Finame)            #-->Read image in path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #-->Convert color to grayscale
    
    print('Clear')
    
    ncnt +=1
    if(img is None):
        print('None image')
        continue
    else:
        print('Process Detect Face')
        faces = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))  #Detect face
        print('clear')
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            facecnt = len(faces)
            print(f"Detected faces: {facecnt}") #-->show number of face(count)
    
            #cv2.imshow('img',img) --> This show original image
    
            height, width = img.shape[:2]

            for (x, y, w, h) in faces:
                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)

                faceimg = img[ny:ny+nr, nx:nx+nr]       #-->Crop image 
                lastimg = cv2.resize(faceimg, (48, 48)) #-->Resize image to 48*48
                n += 1
                #cv2.imshow('img2',lastimg)-->To show image but dispensable
                cv2.imwrite(PathS + 'img'+ str(n) +'.png',lastimg)  #Save and rename in save path
                print(PathS + 'img'+ str(n) +'.png\n')
                
stop = time.time()-start #Finish time
print(f'Time used : {stop:0.2f} sec')




