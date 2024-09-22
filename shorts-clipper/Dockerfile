ARG FUNCTION_DIR="/function"

FROM python:3.12 as build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# Install the function's dependencies
RUN pip install \
    --target ${FUNCTION_DIR} \
        awslambdaric

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}

# Use a slim version of the base Python image to reduce the final image size
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    xz-utils \
    default-libmysqlclient-dev \
    libtool \
    tar \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    fontconfig

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

ENV ENVIRONMENT="production"

RUN curl -L -o /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz && \
    mkdir -p /tmp/ffmpeg && \
    tar xvf /tmp/ffmpeg.tar.xz --strip-components=1 -C /tmp/ffmpeg && \
    mv /tmp/ffmpeg/ffmpeg /usr/local/bin/ && \
    mv /tmp/ffmpeg/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg && \
    chmod +x /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Copy the Futura font file into the Docker image and update the font cache
COPY AntonRegular.ttf /usr/share/fonts/truetype/custom/AntonRegular.ttf
RUN fc-cache -f -v

ENV PRISMA_HOME_DIR="/var/tmp"

COPY ./prisma ./prisma
RUN pip install prisma
RUN prisma generate

COPY . .

# PRODUCTION
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "app.handler" ]

# TESTING / LOCAL
# CMD ["python", "app.py"]
