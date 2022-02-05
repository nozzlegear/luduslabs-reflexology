FROM ubuntu:18.04

MAINTAINER Sindre Henriksen "sid.henriksen@gmail.com"

RUN adduser uwsgi

RUN apt-get update
RUN apt-get update -y &&\
    apt-get install -y python3-pip python3-dev uwsgi uwsgi-plugin-python3

RUN mkdir app && chown -R uwsgi app && chgrp -R uwsgi app
COPY /requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

RUN mkdir /data/reflex

RUN chown -R uwsgi /app &\
    chgrp -R uwsgi /app

EXPOSE 8183

WORKDIR /app/app
CMD [ "uwsgi_python36", "--ini", "uwsgi.ini" ]
