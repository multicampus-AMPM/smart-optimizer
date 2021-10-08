FROM python:3
WORKDIR /home
COPY . .
RUN pip install --no-cache-dir --requirement requirements.txt
CMD ["python", "./optimizer.py"]