FROM python:3.8-slim
WORKDIR /home
COPY ./requirements.txt .
RUN apt-get update && apt-get install libgomp1
RUN pip install --upgrade pip && pip install --no-cache-dir --requirement /home/requirements.txt && pip install tensorflow
COPY ./mlruns ./mlruns
COPY ./optimizer.py .
COPY ./models.py .
COPY ./register.py .
CMD ["python", "./optimizer.py"]