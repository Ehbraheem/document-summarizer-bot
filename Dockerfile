FROM python:3.10

WORKDIR /app

COPY requirements.txt .env .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-u", "server.py"]