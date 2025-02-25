FROM python:3.9-slim

# Install R and related dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    r-recommended \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install R packages
RUN Rscript -e "install.packages(c('ggplot2', 'dplyr', 'readr', 'tidyr', 'scales', 'RColorBrewer'), repos='http://cran.rstudio.com/')"

# Copy application code
COPY . .

# Create a non-root user
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (Streamlit default is 8501)
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "Rpy2.py", "--server.port=8501", "--server.address=0.0.0.0"] 