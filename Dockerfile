# Use an official PyTorch runtime as a parent image
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set the working directory
WORKDIR /ContinuousBernoulliVAE

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of your application's code
COPY . .

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]