# Use the official Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy only Pipfile and Pipfile.lock first (to leverage Docker's caching mechanism)
COPY Pipfile Pipfile.lock ./

# Install dependencies in a virtual environment
RUN pipenv install --deploy --ignore-pipfile

# Copy the rest of the application files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the application
CMD ["pipenv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]