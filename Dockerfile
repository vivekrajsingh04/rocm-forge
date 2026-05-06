FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 7860 (Hugging Face Spaces default)
ENV PORT=7860
EXPOSE 7860

# Command to run the FastAPI app
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
