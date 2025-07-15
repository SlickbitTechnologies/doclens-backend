# Use a lightweight Python image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port (if applicable, default Flask is 5000 or FastAPI is 8000)
EXPOSE 8000

# Start the app
CMD ["python", "main.py"]
