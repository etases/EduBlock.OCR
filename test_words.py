import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['vi'])


def recognize_words(img_input, debug=False) -> "":
    img_resized = cv2.resize(img_input, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    norm_map = np.zeros((img_gray.shape[0], img_gray.shape[1]))
    img_norm = cv2.normalize(img_gray, norm_map, 0, 255, cv2.NORM_MINMAX)
    _, img_threshold = cv2.threshold(img_norm, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if debug:
        cv2.imwrite("debug/resized.jpg", img_resized)
        cv2.imwrite("debug/gray.jpg", img_gray)
        cv2.imwrite("debug/norm.jpg", img_norm)
        cv2.imwrite("debug/threshold.jpg", img_threshold)

    img_ready = img_threshold

    outputs = reader.readtext(img_ready, detail=0)
    text = " ".join(outputs)
    output = ''
    for character in text:
        if character.isalnum() or character == ' ':
            output += character
    output = output.lower()
    return output
