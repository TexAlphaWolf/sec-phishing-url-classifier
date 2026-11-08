FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Scoring needs no third-party package, so the image has none.
RUN pip install . && useradd --create-home urlclf
USER urlclf

# Score URLs with the model bundled in the package, or with one you mount:
#   docker run --rm urlclf score "http://paypal.com.secure-login.example.tk/verify"
#   docker run --rm -v "$PWD/model.json:/m/model.json:ro" urlclf score --model /m/model.json URL
#   docker run --rm -p 127.0.0.1:8080:8080 urlclf serve --host 0.0.0.0
# The image never opens the addresses it is given.
ENTRYPOINT ["urlclf"]
CMD ["--help"]
