FROM python:3.10-slim

WORKDIR /ocr

RUN mkdir debug

COPY . .

RUN python -m venv .venv

RUN .venv/bin/pip install --no-cache-dir --verbose -r requirements.txt

VOLUME ["/ocr/debug"]

EXPOSE 80

CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0"]