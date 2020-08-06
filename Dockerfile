FROM continuumio/miniconda3

WORKDIR /app

# Create the environment
COPY conda.yml .
RUN conda env create -f conda.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "aiap-food-classifier", "/bin/bash", "-c"]

# COPY model weights and Procfile
COPY tensorfood.h5 .

# COPY everything in src folder
COPY src/. .

# Make sure environment is activate
RUN echo "Making sure flask is installed..."
RUN python -c "import flask"

# Code to run with container has started
ENTRYPOINT ["conda", "run", "-n", "aiap-food-classifier", "python", "app.py"]