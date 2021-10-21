FROM python:3.9-slim
WORKDIR /home
COPY ./requirements.txt .
RUN pip install --no-cache-dir --requirement ./requirements.txt
COPY ./mlruns ./mlruns
COPY ./optimizer.py .
CMD ["python", "./optimizer.py"]