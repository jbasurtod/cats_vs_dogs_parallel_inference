FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl iputils-ping

RUN pip install pandas requests

WORKDIR /app

COPY . /app  

RUN mkdir -p /app/results

CMD ["sleep", "infinity"]