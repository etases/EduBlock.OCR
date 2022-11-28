import easyocr
import numpy as np
import torch
from cv2 import cv2

import contours as cont
from model_mnist import init_model, init_transform, init_device
from resize import resize

# Recognize digits with EasyOCR

reader = easyocr.Reader(['en'], detector='DB', recognizer='Transformer')


def recognize_digits_ocr(img_input, debug=False) -> []:
    outputs = reader.readtext(img_input, detail=0, allowlist='.0123456789')
    text = " ".join(outputs)

    if debug:
        print(f"Text: {text}")

    answer = []
    for character in text:
        if character.isdigit():
            answer.append(character)
    return answer


# Recognize digits with OpenCV, MNIST & PyTorch

transform = init_transform()
device = init_device()
model = init_model(device)
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()


def recognize_digits_handwritten(img_input, debug=False):
    img_resized = resize(img_input, width=1000)
    img_copy = img_resized.copy()

    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (25, 10))
    _, img_gray = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    answer = []

    img_canny = cv2.Canny(img_resized, 25, 300)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img_canny, kernel, iterations=1)

    if debug:
        cv2.imwrite("debug/canny.jpg", img_canny)
        cv2.imwrite("debug/dilation.jpg", dilation)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, boundingBoxes = cont.sort_contours(contours, method="left-to-right")

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if debug:
            print(f"Box x:{x} y:{y} w:{w} h:{h}")

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if 50 < w < 200:
            img_new = img_gray[y:y + h, x:x + w]
            img_new = cv2.copyMakeBorder(img_new, 10, 10, 10, 10, borderType=cv2.BORDER_CONSTANT,
                                         value=(255, 255, 255))
            img_new = cv2.resize(img_new, (28, 28), interpolation=cv2.INTER_AREA)

            if debug:
                cv2.imwrite(f"debug/output_{len(answer) + 1}.jpg", img_new)

            torch_input = transform(img_new).unsqueeze(0).to(device)

            output = model(torch_input)
            max_pred = output.argmax(dim=1, keepdim=True)
            predicted_digit = max_pred.item()

            answer.append(predicted_digit)

            if debug:
                print(f"Output: {output}")
                print(f"Answer: {predicted_digit}")

    if debug:
        cv2.imwrite("debug/resized.jpg", img_resized)
        cv2.imwrite("debug/gray.jpg", img_gray)
        cv2.imwrite("debug/blur.jpg", img_blur)
        cv2.imwrite("debug/output.jpg", img_copy)

    return answer


# Main function

def recognize_digits(img_input, debug=False, handwritten=False) -> []:
    if handwritten:
        return recognize_digits_handwritten(img_input, debug)
    else:
        return recognize_digits_ocr(img_input, debug)
