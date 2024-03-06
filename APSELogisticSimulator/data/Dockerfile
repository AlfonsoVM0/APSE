FROM python:3.9-slim-buster

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install flask
COPY . .

WORKDIR /src/service 

EXPOSE 7777

CMD [ "python3", "predictor.py", "--host=0.0.0.0"]