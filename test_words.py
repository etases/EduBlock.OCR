import cv2
import pytesseract as tesseract


def recognize_words(img_input, debug=False) -> "":
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (3, 3))
    img_gray = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if debug:
        cv2.imwrite("debug/blur.jpg", img_blur)
        cv2.imwrite("debug/gray.jpg", img_gray)

    text = tesseract.image_to_string(img_gray, lang="vie", nice=10)
    output = ''
    for character in text:
        if character.isalnum() or character == ' ':
            output += character
    return output
