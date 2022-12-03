import numpy as np
from cv2 import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import contours as cont
from resize import resize
from test_digits import recognize_digits
from test_words import recognize_words

app = FastAPI(docs_url="/swagger")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def imshow(title, image, width=800):
    cv2.imshow(title, resize(image, width=width))
    cv2.waitKey(0)


def threshold(image, invert=False, debug=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding the image
    thresh, img_bin = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert the image
    if invert:
        img_bin = 255 - img_bin

    if debug:
        cv2.imwrite("debug/boxes_threshold.jpg", img_bin)

    return img_bin


def morph(img_bin, kernel, iterations=3):
    img_temp = cv2.erode(img_bin, kernel, iterations=iterations)
    img_lines = cv2.dilate(img_temp, kernel, iterations=iterations)
    return img_lines


def find_boxes(image, debug=False):
    # Defining a kernel length
    kernel_length = np.array(image).shape[1] // 40

    # A vertical kernel of (1 X kernel_length), to detect all the verticle lines.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), to detect all the horizontal lines.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    vertical_lines = morph(image, vertical_kernel)
    horizontal_lines = morph(image, horizontal_kernel)
    boxes = cv2.add(vertical_lines, horizontal_lines)

    if debug:
        cv2.imwrite("debug/boxes_vertical.jpg", vertical_lines)
        cv2.imwrite("debug/boxes_horizontal.jpg", horizontal_lines)

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

    if debug:
        print(text)

    return text


def ocr(img_input, debug=False, reverse=False) -> [[]]:
    # resizing image
    img = resize(img_input, width=2586)
    img_debug = img.copy()

    img_h, img_w, img_c = img.shape

    img_threshold = threshold(img, invert=True, debug=debug)
    boxes = find_boxes(img_threshold, debug)
    boxes = over_draw_boxes(boxes)

    if debug:
        cv2.imwrite("debug/box.jpg", boxes)

    contours, _ = cv2.findContours(boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        for cnt in contours:
            # rectangular contours
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img_debug = cv2.drawContours(img_debug, [box], 0, (0, 0, 255), 3)

    def filter_func(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return (w > 30 and h > 20) and 1 * h < w < 0.8 * img_w

    contours = list(filter(filter_func, contours))

    (contours, boundingBoxes) = cont.sort_contours(contours, method="top-to-bottom")

    elements_per_rows = 4

    cells = []
    row = []
    for idx, cnt in enumerate(contours):
        next_first_row_cell = ((idx + 1) % elements_per_rows) == 0
        row.append(cnt)
        if next_first_row_cell:
            sorted_row_contours, _ = cont.sort_contours(row, method="left-to-right")
            row = []
            if reverse:
                sorted_row_contours = reversed(sorted_row_contours)

            for sorted_index, sorted_cnt in enumerate(sorted_row_contours):
                x, y, w, h = cv2.boundingRect(sorted_cnt)

                # cropped cell img, idx,
                cell = img[y:y + h, x:x + w]
                cells.append(cell)

                # cell mappings (debug)
                if debug:
                    M = cv2.moments(sorted_cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    center = (cx, cy + 20)
                    cv2.putText(img_debug, str(sorted_index), center, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    rows = []
    cols = []
    for idx, img_cell in enumerate(cells):
        name_cell = (idx % elements_per_rows) == 0
        next_first_row_cell = ((idx + 1) % elements_per_rows) == 0

        text = recognize(img_cell, name_cell, debug)
        if len(text) == 0 and not name_cell:
            text = "null"

        cols.append(text.replace("\n", ""))

        if next_first_row_cell:
            rows.append(cols)
            cols = []
    if len(cols) != 0:
        rows.append(cols)

    if debug:
        cv2.imwrite("debug/contour.jpg", img_debug)
        for idx, cell in enumerate(cells):
            cv2.imwrite(f"debug/cell-{str(idx)}.jpg", cell)

    return rows


@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...), debug: bool = False, reverse: bool = False):
    contents = await file.read()
    np_arr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    tables = ocr(img, debug, reverse)
    row_strings = []
    for row in tables:
        row_str = ','.join([str(x) for x in row])
        row_strings.append(row_str)
    return row_strings


@app.post("/recognize/")
async def test_recognize(file: UploadFile = File(...), text: bool = False, debug: bool = True):
    contents = await file.read()
    np_arr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    predict = recognize(img, detect_text=text, debug=debug)
    return predict
