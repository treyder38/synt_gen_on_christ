FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt