FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD python manage.py runserver 0.0.0.0:8000
