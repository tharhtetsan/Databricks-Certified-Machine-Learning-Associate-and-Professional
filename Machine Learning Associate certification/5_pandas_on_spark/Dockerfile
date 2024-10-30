FROM python:3.12-slim

USER root

RUN apt-get update
RUN apt install software-properties-common -y
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y gcc python3-dev openjdk-17-jdk && \
    apt-get clean

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt


# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-arm64

CMD ["python","main.py"]


