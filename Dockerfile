FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*


