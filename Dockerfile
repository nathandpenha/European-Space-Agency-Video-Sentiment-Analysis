# Intel® Distribution of OpenVINO™ toolkit Docker image for Ubuntu* 18.04 LTS
FROM openvino/ubuntu18_dev:2020.4
USER root
WORKDIR /video-sentiment-analysis
COPY data ./data/
COPY docs ./docs/
COPY models ./models/
COPY src ./src/
COPY tests ./tests/
COPY README.md .
COPY requirements.txt .

# Set up the required dependencies
RUN apt update \
 && apt install -y python3.7 python3.7-dev \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# Note: setuptools must be installed first because it is needed to install the other
# depedencies properly.
RUN pip3 install --upgrade pip \
 && pip3 install setuptools \
 && pip3 install -r requirements.txt

# Note: installing dlib can take a while, be patient, your machine has not crashed ;-)

CMD bash
