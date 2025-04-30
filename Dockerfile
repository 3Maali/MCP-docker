FROM python:3.10.6-buster

COPY data data
COPY videos videos

COPY app.py app.py
COPY real.py real.py

COPY requirements-docker.txt requirements-docker.txt

RUN apt-get update
RUN apt-get -y install libgl1

RUN pip install -r requirements-docker.txt

RUN

COPY team.png team.png
COPY qrcode_black.png qrcode_black.png

COPY tawaf.mp4 tawaf.mp4

COPY yolov8s-seg.pt yolov8s-seg.pt
COPY yolov8s.pt yolov8s.pt

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
