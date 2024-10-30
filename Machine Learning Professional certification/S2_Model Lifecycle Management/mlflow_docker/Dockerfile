FROM python:3.8-slim

USER root

ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY ./requirements.txt ./requirements.txt


RUN pip install -r requirements.txt


EXPOSE 8080

CMD mlflow server --host 0.0.0.0 --port 8080