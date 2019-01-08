FROM nvcr.io/nvidia/pytorch:18.11-py3

# Build with:
# docker build -t "pt-challenges" .

# Run with:
# docker run --name "pt-challenges" -it pt-challenges

WORKDIR /app/workflow

CMD [ "/bin/bash" ]

