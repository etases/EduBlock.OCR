import cv2
import numpy as np
import tensorflow as tf
from imutils import contours as cont
import easyocr

model = tf.keras.models.load_model("mnist.h5")
reader = easyocr.Reader(['en'])


def recognize_digits(img_input, debug=False, handwritten=False) -> []:
    if not handwritten:
        outputs = reader.readtext(img_input, detail=0)
        text = " ".join(outputs)
        answer = []
        for character in text:
            if character.isdigit():
                answer.append(character)
        return answer
    else:
        img_resized = cv2.resize(img_input, (1000, 400), interpolation=cv2.INTER_AREA)

        img_copy = img_resized.copy()

        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.blur(img_gray, (25, 10))
        _, img_gray = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        img_canny = cv2.Canny(img_resized, 25, 300)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(img_canny, kernel, iterations=1)

        if debug:
            cv2.imwrite("debug/blur.jpg", img_blur)
            cv2.imwrite("debug/canny.jpg", img_canny)
            cv2.imwrite("debug/dilation.jpg", dilation)

        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, boundingBoxes = cont.sort_contours(contours, method="left-to-right")

        ans = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if debug:
                print(f"Box x:{x} y:{y} w:{w} h:{h}");

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
                    cv2.imwrite(f"debug/output_{len(ans) + 1}.jpg", img_new)

                img_new = np.array(img_new).reshape(1, 28, 28, 1)

                y_pred = model.predict(img_new)
                predicted_digit = np.argmax(y_pred)
                ans.append(predicted_digit)

                if debug:
                    print(f"Answer: {predicted_digit}")

        if debug:
            cv2.imwrite("debug/resized.jpg", img_resized)
            cv2.imwrite("debug/gray.jpg", img_gray)
            cv2.imwrite("debug/output.jpg", img_copy)

        return ans
