
FROM python:3.10-slim


WORKDIR /app


COPY ./monitoring/requirements.txt /app/requirements.txt


RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY ./monitoring/app.py /app/app.py


EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]