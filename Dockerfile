FROM python:3.10-slim

WORKDIR /app

# Copie le code et le modèle
COPY api/ ./api/
COPY models/ ./models/
COPY requirements.txt ./

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.deployment_api:api", "--host", "0.0.0.0", "--port", "8000"]