FROM rust:bookworm as builder
COPY . ./burn
WORKDIR /burn/examples/web-inference
RUN cargo build --release

RUN mkdir libs
RUN find /burn/target/release -name "*.so*" -exec cp {} ./libs \; -print

#FROM debian:bookworm-slim
FROM rust:bookworm
RUN apt update && apt install -y pkg-config libssl-dev libgomp1

COPY --from=builder /burn/target/release/web-inference .
COPY --from=builder /burn/examples/web-inference/libs/* /usr/local/lib
COPY ./addressmultiformat-flant5small-gpttokenizer /datapool/burn_models/addressmultiformat-flant5small-gpttokenizer

ENV LD_LIBRARY_PATH="/usr/local/lib"

CMD ["./web-inference"]
