# Use the official Python 3.12 image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (e.g., git, build tools)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Install pip and build tools
RUN pip install --upgrade pip

# Copy the pyproject.toml and install the project dependencies
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir .

# Copy the rest of the project files into the container
COPY . .

# Set the working directory to src where the scripts are located
WORKDIR /app/src

# Expose port 8888 for Jupyter Notebook (if needed)
EXPOSE 8888

# Default command to keep the container running
CMD ["bash"]