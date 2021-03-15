import numpy as np
import cv2
import time
currentframe=0
# =============== Variable Mouse ==================#
drawing = False
point1 = ()
point2 = ()

Mouse_count = False


# ================================================#
def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing
    global  Mouse_count

    # ----------Mouse 1-------
    if Mouse_count == False:
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing is False:
                drawing = True
                point1 = (x, y)
            # else:
            # drawing = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing is True:
                point2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            Mouse_count = True
#---cap = cv2.VideoCapture("K:\EDD\EDI dec2k20\Video dataset/ankita.mp4")
#----cap = cv2.VideoCapture("C:/Users/kshirsagar/Downloads/youtube.mp4")
cap = cv2.VideoCapture("K:\EDD\EDI dec2k20\Video dataset/merged_day.avi")
#cap = cv2.VideoCapture('K:\EDD\EDI dec2k20/traffic_violation_video.wmv')


cv2.namedWindow("Detecion Car")
cv2.setMouseCallback("Detecion Car", mouse_drawing)

while True:
    ret, img = cap.read()

    car_cascade = cv2.CascadeClassifier('C:/Users/kshirsagar/Desktop/cars.xml')

    if point1 and point2:

        # Rectangle marker
        r1 = cv2.rectangle(img, point1, point2, (100, 50, 200), 5)
        frame_ROI = img[point1[1]:point2[1], point1[0]:point2[0]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cimg = img


        # converting to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # color range--Giving range for red, green and yellow
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        lower_yellow = np.array([15, 150, 150])
        upper_yellow = np.array([35, 255, 255])

        # to perform basic thresholding operations to detecting specific color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        maskg = cv2.inRange(hsv, lower_green, upper_green)
        masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
        maskr = cv2.add(mask1, mask2)
        size = img.shape

        # used Hough Transform to draw circle on specific color of signal
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                     param1=50, param2=10, minRadius=0, maxRadius=30)

        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                     param1=50, param2=10, minRadius=0, maxRadius=30)

        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                     param1=50, param2=5, minRadius=0, maxRadius=30)

        # traffic light detect
        r = 5
        bound = 4.0 / 10
        if r_circles is not None:
            r_circles = np.uint16(np.around(r_circles))

            for i in r_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += maskr[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(maskr, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                    cv2.putText(cimg, 'RED', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    if drawing is False:
                        # convert video into gray scale of each frames
                        ROI_grayscale = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
                        # detect cars in the video
                        cars_ROI = car_cascade.detectMultiScale(ROI_grayscale, 1.1, 1)
                        for (x, y, w, h) in cars_ROI:
                            cv2.rectangle(frame_ROI, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            name = 'K:\EDD\EDI dec2k20/data/frame' + str(currentframe) + '.jpg'
                            print('creating...' + name)
                            cv2.imwrite(name, img)
                            currentframe += 1

        if g_circles is not None:
            g_circles = np.uint16(np.around(g_circles))

            for i in g_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += maskg[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 100:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(maskg, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                    cv2.putText(cimg, 'GREEN', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if y_circles is not None:
            y_circles = np.uint16(np.around(y_circles))

            for i in y_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += masky[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(masky, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                    cv2.putText(cimg, 'YELLOW', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # ------------------Detect car ROI-------------------#

        # ------------------Detect car ROI-------------------#


    cv2.imshow("Detecion Car", img)

    if cv2.waitKey(33)==27:
        break



cap.release()
cv2.destroyAllWindows()