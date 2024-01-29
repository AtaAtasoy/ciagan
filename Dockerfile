FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y wget git build-essential python3 python3-pip python3-opencv ffmpeg libsm6 libxext6
RUN pip install opencv-python

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt