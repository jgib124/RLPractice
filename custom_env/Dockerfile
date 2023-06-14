# syntax=docker/dockerfile:1

# Base image
FROM python:3.8-slim-buster

# Work Environment
WORKDIR /RLPractice/custom_env

# Copy the requirements.txt file
COPY requirements.txt requirements.txt

# Install the requirements
RUN pip3 install -r requirements.txt

# Dependencies installed, add source code to image
COPY . .

# Run the app
CMD [ "python3", "ppo_agent.py" ]
