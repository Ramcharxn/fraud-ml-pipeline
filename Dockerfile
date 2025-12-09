FROM public.ecr.aws/docker/library/python:3.10-slim

# Work directory inside the container
WORKDIR /app

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        sagemaker \
        boto3 \
        lightgbm \
        scikit-learn \
        pandas \
        numpy

# Copy your training code into the image
COPY start_training.py fraud_sagemaker.py ./

# When the container runs, it will kick off training
ENTRYPOINT ["python", "start_training.py"]
