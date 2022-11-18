import uuid
from fastapi import FastAPI
import cv2
import imutils
from imutils import contours as cont
import numpy as np
import csv
import pytesseract as tesseract
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
import os
from random import randint
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse, HTMLResponse, FileResponse

app = FastAPI()

def sort_contours(cnts, method):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def imshow(title, image, width=800):
    cv2.imshow(title, imutils.resize(image, width=width))
    cv2.waitKey(0)


def threshold(image, invert=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Thresholding the image
    thresh, img_bin = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert the image
    if invert: img_bin = 255-img_bin
    return img_bin

def morph(img_bin, kernel, iterations=3):
    img_temp = cv2.erode(img_bin, kernel, iterations=iterations)
    img_lines = cv2.dilate(img_temp, kernel, iterations=iterations)
    return img_lines

def find_boxes(image):
    #convert binary image
    img_bin = threshold(image, invert=True)

    # Defining a kernel length
    kernel_length = np.array(img_bin).shape[1]//40

    # A verticle kernel of (1 X kernel_length), to detect all the verticle lines.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), to detect all the horizontal lines.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    verticle_lines = morph(img_bin, verticle_kernel)
    horizontal_lines = morph(img_bin, horizontal_kernel)
    boxes = cv2.add(verticle_lines, horizontal_lines)

    return boxes

def over_draw_boxes(img_bin):
    minLineLength=100
    lines = cv2.HoughLinesP(image=img_bin,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)
    for i in range(lines.shape[0]):
        cv2.line(img_bin, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 2, cv2.LINE_AA)

    return img_bin


def ocr():
    # im = Image.open(urlopen("Table_Ex.jpg"))
    img = cv2.imread("Table_Ex.jpg")

    #resizing image
    img = imutils.resize(img, width=2586)
    img_original = img.copy()

    boxes = find_boxes(img)
    boxes = over_draw_boxes(boxes)


    contours, _ = cv2.findContours(boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #_, contours, _ = cv2.findContours(boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    (contours, boundingBoxes) = cont.sort_contours(contours, method="left-to-right")
    (contours, boundingBoxes) = cont.sort_contours(contours, method="top-to-bottom")

    idx = 0
    images = []
    rows = []
    i = 0
    j = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if (w > 30 and h > 20) and w > 1*h:
            #rectangular contours
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img = cv2.drawContours(img, [box], 0, (0,0,255), 3)


            #cell mappings
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = (cx, cy+20)
            if idx!=0:
                cv2.putText(img, str(idx), center, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
                cv2.imwrite("cropped/0.jpg", img)

            #cropped cell
            cell = img_original[y:y+h, x:x+w]
            h, w, c = img.shape
            rate = h / w
            if idx == 0:
                cv2.imwrite("cropped/craft-"+str(idx)+".jpg", cell)
            else:
                resize = cv2.resize(cell, None, fx=0.5, fy=0.5,
                                    interpolation=cv2.INTER_LINEAR)
                cv2.imwrite("cropped/row-"+str(i)+"-col-"+str(j)+".jpg",resize)
                j += 1
            # imshow("image", img)
            idx+=1

            if j >= 4:
                j = 0
                i += 1

    a = i
    fig, axs = plt.subplots(a, 4)
    fig.suptitle('Images in detection')

    rows = []
    images = []

    for i in range(a):
        cols = []
        for j in range(4):

            img = cv2.imread(f"cropped/row-{i}-col-{j}.jpg")

            h,w,c = img.shape
            rate = h/w

            # if (j == 0):
            #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #     gray = cv.threshold(gray, 0, 255,cv.THRESH_BINARY |cv.THRESH_OTSU)[1]
            #     filename = "{}.png".format(os.getpid())
            #     cv.imwrite(filename, gray)
            #     images.append(axs[i, j].imshow(img))
            #     text = tesseract.image_to_string(Image.open(filename),lang='vie')
            #     os.remove(filename)
            # else:



            resized_img = cv2.resize(img, None, fx=rate, fy=rate,
                                    interpolation=cv2.INTER_LINEAR)

            sharped_img = cv2.addWeighted(
                img, 4, cv2.blur(img, (25, 25)), -4, 350)

            # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            # filename = "{}.png".format(os.getpid())
            # cv.imwrite(filename, gray)
            #
            # images.append(axs[i, j].imshow(sharped_img))
            #
            # text = tesseract.image_to_string(Image.open(filename),config='--psm 12 --oem 3', lang='vie')
            # os.remove(filename)

            if (j == 0):
                # images.append(axs[i, j].imshow(sharped_img))
                text = tesseract.image_to_string(
                sharped_img, config='--psm 6', lang="vie")
            else:
                # Convert to the gray-scale
                gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Down-sample
                gry = cv2.resize(gry, (0, 0), fx=0.45, fy=0.45)
                # images.append(axs[i, j].imshow(sharped_img))
                text = tesseract.image_to_string(gry)

            if (len(text) == 0 and j > 0):
                text = "null"

            cols.append(text.replace("\n", ""))
        cols.reverse()
        rows.append(cols)

    with open("results.csv", "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"Table_Ex.jpg"
    contents = await file.read()

    with open(f"{file.filename}", "wb") as f:
        f.write(contents)

    ocr()

    path = f"results.csv"
    return FileResponse(path)