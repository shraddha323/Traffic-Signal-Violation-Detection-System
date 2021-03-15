import numpy as np
import cv2
import time

#cascade_src='C:/Users/kshirsagar/PycharmProjects/EDI/classifier/cascade.xml'
video_src='C:/Users/kshirsagar/Downloads/WhatsApp Video 2020-10-03 at 9.51.34 PM.3gpp'
#video_src='C:/Users/kshirsagar/Desktop/car2.jpg'
#video_src='K:/EDD/EDI dec2k20/cars.mp4'
cap=cv2.VideoCapture(video_src)

car_cascade=cv2.CascadeClassifier('cascade.xml')
#car_cascade=cv2.CascadeClassifier('C:/Users/kshirsagar/Downloads/cascade.xml')
currentframe=2
img=cv2.imread('C:/Users/kshirsagar/Desktop/car2.jpg')
while (type(img)!=type(None)):
   # ret,img=cap.read()
    img = cv2.imread('K:/EDD/EDI dec2k20/p/' + str(currentframe) + '.jpeg')
    #img = cv2.imread('C:/Users/kshirsagar/PycharmProjects/data-p/' + str(currentframe) + '.jpeg')
    #cv2.imshow('img', img)
    #img = cv2.imread('C:/Users/kshirsagar/Desktop/car2.jpg')

    if(type(img)==type(None)):
        break

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cars=car_cascade.detectMultiScale(gray,1.1,1)

    plate = None
    """for c in cnts:
        perimeter = cv2.arcLength(c, True)
        edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(edges_count) == 4:
            x, y, w, h = cv2.boundingRect(c)
            plate1 = image[y:y + h, x:x + w]
            break
    """
    for(x,y,w,h)in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

        plate = img[y:y + h, x:x + w]
        #cv2.imwrite('plate.png', plate)


        #cv2.imshow('plate', plate)

        cv2.imshow('video', img)
        name = 'K:\EDD\EDI dec2k20/data2/plate' + str(currentframe) + '.jpg'

        cv2.imwrite(name, plate)
        print('creating...' + name)
    time.sleep(1)
    currentframe += 1

    if cv2.waitKey(33)==27:
        break

import glob
import os
import pandas as pd

files_dir = 'K:\EDD\EDI dec2k20/data2'  # here should be path to your directory with images

files = glob.glob(os.path.join(files_dir, '*'))

df = pd.DataFrame(columns=['name', 'format', 'no.plates'])

for i, full_filename in enumerate(files):
    filename = os.path.basename(full_filename)

    name, format_ = filename.split('.', 1)

    format_ = os.path.splitext(format_)  # remove file extension from format_

    hyperlink = '=HYPERLINK("file:/{}")'.format(full_filename)
    df.loc[i] = [name, format_, hyperlink]

# Here, images will saved in the excel sheet.
df.to_excel('K:\EDD\EDI dec2k20\output_file.xlsx', index=False)

cv2.waitKey()  # to hold image
cv2.destroyAllWindows()


