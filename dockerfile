FROM python:3.10-slim

WORKDIR /app

# copy requirements.txt file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy remaining files into the container
COPY . .

# Expose default port for streamlit
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]