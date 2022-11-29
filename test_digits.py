import easyocr

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


def recognize_digits(img_input, debug=False) -> []:
    return recognize_digits_ocr(img_input, debug)
