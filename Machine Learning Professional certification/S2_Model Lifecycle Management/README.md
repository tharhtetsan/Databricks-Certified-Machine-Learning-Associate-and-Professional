```bash
# run database
docker compose up

# run mlflow
mlflow ui  --host 127.0.0.1 --port 8080 --backend-store-uri postgresql+psycopg2://postgres:tharhtet@localhost:5432/test_db
```