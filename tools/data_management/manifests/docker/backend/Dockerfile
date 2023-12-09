FROM python:3.10-slim-bullseye
WORKDIR /opt/app
ENV PYTHONPATH "${PYTHONPATH}:/opt/app"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY src ./src

ENTRYPOINT ["python", "src/backend", "serve"]