# Reference container for ds4 on DGX Spark / GB10 (aarch64).
#
# The one rule: the GB10 build target is `make cuda-spark`.  A generic
# `make cuda CUDA_ARCH=...` build boots and serves on a GB10, but without
# the DGX Spark configuration (sm_121a SASS + the Spark HBM weight cache)
# it runs at roughly a third of the prefill speed and half the decode
# speed (field report: NVIDIA forum 378855 post 48, 370 tok/s prefill /
# 15 decode vs ~1000 / 30+).  ds4-server prints a boot advisory when a
# non-spark binary lands on a GB10, and this Dockerfile exists so the
# advisory never has to fire.
#
# Build (no GPU needed -- the arch is set explicitly, from the repo root):
#
#   docker build -t ds4:spark -f docker/Dockerfile .
#
# Run.  The model stays OUTSIDE the image, bind-mounted from the host:
#
#   docker run --gpus all --ipc=host --ulimit memlock=-1 \
#     -v /path/to/models:/models -p 8000:8000 ds4:spark \
#     ./ds4-server -m /models/ds4flash.gguf -c 32768 --host 0.0.0.0
#
# MEMORY: the container must see the FULL host memory.  ds4 mmaps the
# ~155 GiB weight file and serves cold expert reads straight from the
# host page cache, so never set a container memory limit (-m/--memory,
# or mem_limit in compose): mmap'd page-cache pages are charged to the
# container's cgroup, and under a cap the kernel evicts exactly those
# weight pages -- every later read goes back to disk and the server
# crawls.  Bind-mounting the model (rather than baking it into the
# image) keeps that page cache shared with the host and warm across
# container restarts.
#
# The base tag needs a CUDA 13.x devel image (sm_121 support arrived in
# CUDA 13); bump the tag to match the host driver if needed.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ds4
ENV PATH="/root/.cargo/bin:${PATH}"
COPY rust-toolchain.toml .
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --default-toolchain none \
    && rm /tmp/rustup-init.sh \
    && rustup show active-toolchain

COPY . .

# cuda-spark: -gencode arch=compute_121a,code=sm_121a (+DS4_CUDA_HAVE_MXF4).
# Purely an arch alias -- weight residency is a runtime policy
# (DS4_WEIGHT_RESIDENCY), not a compile fork.  Verify with
# `cuobjdump ds4_cuda.o | grep sm_121a` if in doubt.
RUN make -j"$(nproc)" cuda-spark

EXPOSE 8000
CMD ["./ds4-server", "-m", "/models/ds4flash.gguf", "--host", "0.0.0.0"]
