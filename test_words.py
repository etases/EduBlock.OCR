import easyocr

reader = easyocr.Reader(['vi', 'en'])


def recognize_words(img_input, debug=False) -> "":
    # img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.blur(img_gray, (3, 3))
    # img_gray = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #
    # if debug:
    #     cv2.imwrite("debug/blur.jpg", img_blur)
    #     cv2.imwrite("debug/gray.jpg", img_gray)

    outputs = reader.readtext(img_input, detail=0)
    text = " ".join(outputs)
    output = ''
    for character in text:
        if character.isalnum() or character == ' ':
            output += character
    return output
