FROM alpine:latest AS builder

RUN apk update \
    && apk add --no-cache g++ make cmake boost-dev curl-dev

WORKDIR /home/app/
RUN mkdir -p /home/app/coordinator
WORKDIR /home/app/coordinator/
COPY . .

RUN mkdir -p build
WORKDIR /home/app/coordinator/build

RUN cmake ..
RUN make


FROM alpine:latest
RUN apk update \
    && apk add --no-cache g++ curl

RUN mkdir -p /home/app/coordinator/build
COPY --from=builder /home/app/coordinator/build /home/app/coordinator/build

# Add non root user
RUN addgroup -S app && adduser app -S -G app

RUN chown -R app /home/app/coordinator/build

USER app

WORKDIR /home/app/coordinator/build
CMD ["./src/coordinator /dev/shm/"]