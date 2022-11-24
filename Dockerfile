FROM python:3.10.2-slim

WORKDIR /

RUN mkdir /debug
COPY /main.py /
COPY /test_digits.py /
COPY /test_words.py /
COPY /mnist.h5 /
COPY /requirements.txt /

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]