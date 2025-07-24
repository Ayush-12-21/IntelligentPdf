FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install pymupdf==1.26.3

CMD ["python", "main.py"]
