FROM python:3.6

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    jq \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN git clone https://github.com/usc-isi-i2/datamart-wikifier
RUN cd /app/datamart-wikifier/wikifier && python -m venv wiki_env && /bin/bash -c "source wiki_env/bin/activate" && pip install -r requirements.txt

EXPOSE 8396
WORKDIR /app/datamart-wikifier/wikifier
RUN chmod +x runTfidfServer.sh
CMD  /bin/bash -c "./runTfidfServer.sh"
