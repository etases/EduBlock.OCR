import cv2
import easyocr

reader = easyocr.Reader(['vi'])


def recognize_words(img_input, debug=False) -> "":
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    _, img_threshold = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if debug:
        cv2.imwrite("debug/gray.jpg", img_gray)
        cv2.imwrite("debug/threshold.jpg", img_threshold)

    img_ready = img_threshold

    outputs = reader.readtext(img_ready, detail=0)
    text = " ".join(outputs)
    output = ''
    for character in text:
        if character.isalnum() or character == ' ':
            output += character
    return output
