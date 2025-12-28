# 1. Base Image: Start with a lightweight Python 3.10 environment
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first (for better caching)
COPY requirements.txt .

# 4. Install libraries
# We use --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your source code into the container
COPY src/ ./src/

# 6. Copy the trained model into the container
# In a CI/CD pipeline, you might pull this from cloud storage instead
COPY models/ ./models/

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Define the command to start the server
# host 0.0.0.0 is crucial for Docker containers to be accessible from outside
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]