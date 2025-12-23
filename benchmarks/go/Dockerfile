ARG TINYGO_VERSION=0.40.0
ARG GO_VERSION=1.25

# Stage 1: TinyGo builds
FROM tinygo/tinygo:${TINYGO_VERSION} AS builder-tinygo

WORKDIR /usr/src

# Copy all benchmark source and build script
COPY --chown=tinygo:tinygo json json
COPY --chown=tinygo:tinygo regex regex
COPY --chown=tinygo:tinygo build.sh .

# Create output directory and build with TinyGo
USER root
RUN mkdir -p /benchmark/tinygo && chown tinygo:tinygo /benchmark/tinygo

USER tinygo
RUN GO_COMPILER=tinygo OUTPUT_DIR=/benchmark/tinygo ./build.sh

# Stage 2: Standard Go builds
FROM golang:${GO_VERSION} AS builder-go

WORKDIR /usr/src

# Copy all benchmark source and build script
COPY json json
COPY regex regex
COPY build.sh .

# Create output directory and build with standard Go
RUN mkdir -p /benchmark/go
RUN GO_COMPILER=go OUTPUT_DIR=/benchmark/go ./build.sh

# Stage 3: Combine both builds into final image
FROM busybox:latest AS final

# Copy TinyGo builds
COPY --from=builder-tinygo /benchmark/tinygo /benchmark/

# Copy Go builds and rename them
COPY --from=builder-go /benchmark/go /tmp/go-builds
RUN cd /tmp/go-builds && \
    for f in tinygo-*.wasm; do \
    mv "$f" "/benchmark/go-${f#tinygo-}"; \
    done && \
    rm -rf /tmp/go-builds
