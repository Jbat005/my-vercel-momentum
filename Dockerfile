# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask APIs
EXPOSE 8080

# Start both APIs using a process manager
CMD ["sh", "-c", "waitress-serve --host 0.0.0.0 --port 8080 momentum_api:app & waitress-serve --host 0.0.0.0 --port 8081 monte_carlo_api:app"]
