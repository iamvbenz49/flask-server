FROM python:3.10-slim-buster


WORKDIR /python-docker

COPY requirements.txt requirements.txt

RUN  pip install -r requirements.txt

COPY . .
COPY .env . 

EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]

