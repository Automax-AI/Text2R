FROM python:3.9-slim

# Install R and related dependencies with additional system dependencies for R packages
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
    # Additional dependencies for R packages
    libgit2-dev \
    libssh2-1-dev \
    libsodium-dev \
    libicu-dev \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Copy .env file
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install R packages with error handling
RUN Rscript -e "options(repos = c(CRAN = 'https://cloud.r-project.org/')); \
    install.packages(c('ggplot2', 'dplyr', 'readr', 'tidyr', 'scales', 'RColorBrewer')); \
    if (!require('lubridate', quietly = TRUE)) install.packages('lubridate'); \
    if (!require('viridis', quietly = TRUE)) install.packages('viridis'); \
    if (!require('plotly', quietly = TRUE)) install.packages('plotly'); \
    if (!require('ggthemes', quietly = TRUE)) install.packages('ggthemes'); \
    if (!require('gridExtra', quietly = TRUE)) install.packages('gridExtra'); \
    if (!require('reshape2', quietly = TRUE)) install.packages('reshape2'); \
    if (!require('forcats', quietly = TRUE)) install.packages('forcats'); \
    if (!require('readxl', quietly = TRUE)) install.packages('readxl'); \
    if (!require('knitr', quietly = TRUE)) install.packages('knitr'); \
    if (!require('broom', quietly = TRUE)) install.packages('broom'); \
    if (!require('modelr', quietly = TRUE)) install.packages('modelr'); \
    if (!require('data.table', quietly = TRUE)) install.packages('data.table');"

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