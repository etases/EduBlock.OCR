import csv
import os
import uuid

import cv2
import imutils
import numpy as np

from matplotlib import pyplot as plt
from fastapi import FastAPI, File, UploadFile
from imutils import contours as cont
from test_mnist import recognize_digits
from test_words import recognize_words

app = FastAPI()


def imshow(title, image, width=800):
    cv2.imshow(title, imutils.resize(image, width=width))
    cv2.waitKey(0)


def threshold(image, invert=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding the image
    thresh, img_bin = cv2.threshold(
        image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert the image
    if invert:
        img_bin = 255 - img_bin
    return img_bin


def morph(img_bin, kernel, iterations=3):
    img_temp = cv2.erode(img_bin, kernel, iterations=iterations)
    img_lines = cv2.dilate(img_temp, kernel, iterations=iterations)
    return img_lines


def find_boxes(image):
    # convert binary image
    img_bin = threshold(image, invert=True)

    # Defining a kernel length
    kernel_length = np.array(img_bin).shape[1] // 40

    # A vertical kernel of (1 X kernel_length), to detect all the verticle lines.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), to detect all the horizontal lines.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_length, 1))

    vertical_lines = morph(img_bin, vertical_kernel)
    horizontal_lines = morph(img_bin, horizontal_kernel)
    boxes = cv2.add(vertical_lines, horizontal_lines)

    return boxes


def over_draw_boxes(img_bin):
    min_line_length = 100
    lines = cv2.HoughLinesP(image=img_bin, rho=1, theta=np.pi / 180, threshold=110, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=80)
    for i in range(lines.shape[0]):
        cv2.line(img_bin, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 2,
                 cv2.LINE_AA)

    return img_bin


def recognize(img, detect_text, debug=False):
    text = ""
    if detect_text:
        text = recognize_words(img, debug)
    else:
        digits = recognize_digits(img, debug)
        size = len(digits)
        if size == 1:
            text = f"{str(digits[0])}.0"
        elif size == 2:
            text = f"{str(digits[0])}.{str(digits[1])}"
        elif size >= 3:
            text = f"{str(digits[0])}.{str(digits[2])}"

    print(text)
    return text


def ocr(filename, csvName, debug=False, reverse=bool):
    # im = Image.open(urlopen("Table_Ex.jpg"))
    img = cv2.imread(filename)
    # img = file

    # resizing image
    img = imutils.resize(img, width=2586)
    img_original = img.copy()

    img_h, img_w, img_c = img.shape

    boxes = find_boxes(img)
    boxes = over_draw_boxes(boxes)

    contours, _ = cv2.findContours(boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def filter_func(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return (w > 30 and h > 20) and 1 * h < w < 0.8 * img_w
    contours = list(filter(filter_func, contours))

    (contours, boundingBoxes) = cont.sort_contours(contours, method="left-to-right")
    (contours, boundingBoxes) = cont.sort_contours(contours, method="top-to-bottom")

    if reverse:
        contours = reversed(contours)

    images = []
    rows = []
    i = 0
    j = 0
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # rectangular contours
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

        # cell mappings
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = (cx, cy + 20)
        cv2.putText(img, str(idx), center, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imwrite("cropped/0.jpg", img)

        # cropped cell img, idx,
        cell = img_original[y:y + h, x:x + w]
        resize = cv2.resize(cell, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite("cropped/row-" + str(i) + "-col-" + str(j) + ".jpg", resize)

        j += 1
        if j >= 4:
            j = 0
            i += 1

    a = i
    fig, axs = plt.subplots(a, 4)
    fig.suptitle('Images in detection')

    rows = []
    # images = []

    for i in range(a):
        cols = []
        for j in range(4):
            img = cv2.imread(f"cropped/row-{i}-col-{j}.jpg")
            text = recognize(img, j == 0, debug)

            if len(text) == 0 and j > 0:
                text = "null"

            cols.append(text.replace("\n", ""))
        # cols.reverse()
        rows.append(cols)

    with open(csvName, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...), debug: bool = False, reverse: bool = False):
    csvName = f"{uuid.uuid4()}.csv"
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    with open(f"{file.filename}", "wb") as f:
        f.write(contents)

    ocr(file.filename, csvName, debug, reverse)

    # data = pandas.read_csv(csvName)
    # path = f"results.csv"

    data = []
    with open(csvName, newline='', encoding='utf-8', errors='ignore') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            data.append(','.join(row))
    list = [x for x in data if x]

    if os.path.exists(file.filename):
        os.remove(file.filename)
        print("The img has been deleted")
    else:
        print("The img does not exist")

    # if os.path.exists(csvName):
    #     os.remove(csvName)
    #     print("The csv has been deleted")
    # else:
    #     print("The csv does not exist")

    return list


@app.post("/recognize/")
async def test_recognize(file: UploadFile = File(...), text: bool = False, debug: bool = True):
    contents = await file.read()
    np_arr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    predict = recognize(img, detect_text=text, debug=debug)
    return predict
