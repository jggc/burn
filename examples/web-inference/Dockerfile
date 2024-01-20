# Stage 1: Build the application
FROM rust:latest as builder

# Create a new empty shell project
RUN USER=root cargo new --bin rustyparser
WORKDIR /rustyparser

# Copy our manifests
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml

# Build only the dependencies to cache them
RUN cargo build --release
RUN rm src/*.rs

# Now that the dependencies are built, copy your source code
COPY ./src ./src

# Build for release
RUN rm ./target/release/deps/rustyparser*
RUN cargo build --release

# Stage 2: Setup the runtime environment
FROM debian:buster-slim

# Copy the build artifact from the build stage
COPY --from=builder /rustyparser/target/release/rustyparser .

# Set the startup command to run your binary
CMD ["./rustyparser"]

