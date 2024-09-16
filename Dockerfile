FROM python:3.10

WORKDIR /app

COPY requirements.txt .env .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "server.py"]