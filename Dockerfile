# Phase 9 (D-029) — sepsis-app runtime image.
# Base matches D-018 (Python 3.10). Package pins in requirements.txt are
# load-compatibility-critical (scikit-learn==1.6.1 / numpy layout) — do not
# change them here.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install pinned dependencies first (layer cache-friendly).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application, including the frozen model artifact:
# Backend/Model/hgb_sepsis_model.joblib (included via the Backend/ copy).
COPY Backend/ ./Backend/

EXPOSE 8000

# Non-root runtime user.
RUN adduser --disabled-password --gecos "" sepsisapp
USER sepsisapp

CMD ["uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "8000"]