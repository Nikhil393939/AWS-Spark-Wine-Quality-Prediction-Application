# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04


# Install dependencies
RUN apt-get update && \
    apt-get install -y openjdk-16-jdk wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-16-openjdk-arm64
ENV PATH=$PATH:$JAVA_HOME/bin


# Download and install Apache Spark
RUN wget -qO- https://downloads.apache.org/spark/spark-3.4.3/spark-3.4.3-bin-hadoop3.tgz | tar xvz -C /opt && \
    mv /opt/spark-3.4.3-bin-hadoop3 /opt/spark

    
# Set Spark environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Create a directory for the application
WORKDIR /app

# copy necessary content to docker app
COPY jar_and_models /app/jar_and_models
COPY dataset /app/dataset

ENTRYPOINT ["spark-submit", "--class", "modelPrediction", "jar_and_models/model_pred.jar"]