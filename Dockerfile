## Pull base image
FROM ubuntu:focal
#

ENV PYTHONUNBUFFERED=1


RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3-pip git

COPY . /app
WORKDIR /app

RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.8 /usr/bin/python
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3

RUN pip install --upgrade pip

#RUN pip install --default-timeout=1000 -r requirements.txt
RUN pip install --default-timeout=1000 -r requirements.txt --cache-dir ~/.pip-cache


EXPOSE 8000
CMD ["python", "main.py"]
