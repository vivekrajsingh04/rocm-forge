FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (Render/HF use 10000 or 7860 usually, we'll expose 8000 but make it configurable)
ENV PORT=8000
EXPOSE 8000

# Command to run the FastAPI app
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
