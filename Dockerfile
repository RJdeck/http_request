FROM nvidia/cuda:11.3.0-base-ubuntu20.04

WORKDIR /app
COPY . .
RUN apt update && apt install -y python3-pip
RUN apt install -y git
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
RUN pip3 install -r requirements.txt --no-cache-dir

EXPOSE 5000

CMD ["python3", "http_request.py"]