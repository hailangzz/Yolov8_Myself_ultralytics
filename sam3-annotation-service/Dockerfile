FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime


WORKDIR /app


ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive


#######################################
# System dependencies
#######################################

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/



#######################################
# Python dependencies
#######################################

COPY requirements.txt .


RUN pip install --no-cache-dir \
    -r requirements.txt



#######################################
# Application
#######################################
COPY CLIP ./CLIP

RUN pip install ./CLIP

COPY app ./app

COPY models ./models

COPY config ./config



#######################################
# SAM3 resources
#######################################

# 如果你的 tokenizer 文件单独管理
#COPY assets ./assets



#######################################
# Service port
#######################################

EXPOSE 9001



#######################################
# Start FastAPI
#######################################



CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001"]