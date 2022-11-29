# EduBlock.OCR

## Local

### Setup
```shell
python -m venv .venv
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Run
```shell
uvicorn main:app --reload
```

## Docker

### Build Images
```shell
docker build -t edublock-ocr .
```

### Run Container

#### CPU
```shell
docker run -d --name edublock-ocr-ctn -p 80:80 edublock-ocr
```

#### GPU
> Requires `nvidia-docker` for Linux
```shell
docker run -d --gpus all --name edublock-ocr-ctn -p 80:80 edublock-ocr
```