FROM python:3.11-slim
WORKDIR /workspace

RUN python -m pip install --upgrade pip

COPY requirements/base.txt /req/base.txt
COPY requirements/dash.txt /req/dash.txt
RUN pip install --no-cache-dir -r /req/base.txt -r /req/dash.txt

COPY frontend /workspace/frontend

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV PORT=8050
EXPOSE $PORT

CMD python frontend/ui/dash_app.py
