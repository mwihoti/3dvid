# 3dvid studio - container image for IBM Cloud Code Engine, AWS EC2, or any Docker host.
#   CPU:  docker build -t 3dvid .
#   GPU:  docker build --build-arg TORCH_INDEX=cu128 -t 3dvid .   (run with --gpus all)
# Model weights are baked in at build time so cold starts don't re-download.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# nunif (iw3) pinned to the same commit as the git submodule. Cloned here rather
# than COPY'd so the build does not depend on how the CI checks out submodules.
ARG NUNIF_COMMIT=d23721f
RUN git clone --filter=blob:none https://github.com/nagadomi/nunif.git nunif \
    && cd nunif && git checkout -q ${NUNIF_COMMIT} && rm -rf .git

# torch matching nunif's pinned version (2.7.1). TORCH_INDEX=cpu (default) or cu128 for
# NVIDIA GPUs - the CUDA wheels bundle their own runtime, so the slim base image is fine;
# the host only needs the driver + nvidia-container-toolkit (docker run --gpus all).
ARG TORCH_INDEX=cpu
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
    && pip install -r nunif/requirements.txt
COPY requirements-stylize.txt .
RUN pip install -r requirements-stylize.txt

# --- bake model weights -------------------------------------------------------
# depth model (Depth-Anything-V2-Small) via torch.hub -> nunif/iw3/pretrained_models/hub
RUN cd nunif && python -c "from iw3.depth_model_factory import create_depth_model as c; c('Any_V2_S').load(gpu=-1)"
# stereo warp weights (row_flow, depth_aa) are fetched lazily by iw3.cli: run it once on a tiny clip
RUN ffmpeg -v error -f lavfi -i testsrc=size=64x64:rate=10 -t 0.5 -pix_fmt yuv420p /tmp/t.mp4 \
    && cd nunif && python -m iw3.cli --gpu -1 --depth-model Any_V2_S --half-sbs -i /tmp/t.mp4 -o /tmp/o.mp4 --yes \
    && rm -f /tmp/t.mp4 /tmp/o.mp4
# person mask model (rembg u2net_human_seg -> /root/.rembg)
RUN python -c "from rembg import new_session; new_session('u2net_human_seg')"

COPY stylize.py serve.py convert3d.sh ./
COPY web ./web

# runtime layout: everything mutable lives under DATA_DIR (mount a volume / COS bucket here)
ENV HOST=0.0.0.0 PORT=8080 DATA_DIR=/app/data STYLIZE_PYTHON=python3
RUN mkdir -p /app/data
VOLUME ["/app/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')" || exit 1
CMD ["python", "serve.py"]
