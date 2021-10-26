FROM python:3.9-slim
WORKDIR /home
COPY ./requirements.txt .
RUN apt-get update && apt-get install libgomp1
RUN pip install --no-cache-dir --requirement ./requirements.txt
COPY ./mlruns ./mlruns
COPY ./data ./data
COPY ./optimizer.py .
COPY ./models.py .
CMD ["python", "./optimizer.py"]