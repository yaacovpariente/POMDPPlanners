# Use the base image
FROM pomdp-planners-base:latest

# Copy the project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set the entrypoint
ENTRYPOINT ["python"] 