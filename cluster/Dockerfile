# Image for the hessian.AI 42 cluster (MLDE / Determined 0.35.0).
#
# WHY A CUSTOM IMAGE AT ALL: the stock Determined images are py-3.8-pytorch-1.12, and the
# compute nodes forbid installing anything at run time ("apt install, pdsh, apt-get won't
# work"). Qwen3 / Llama-3.1 / Gemma-2 reward models need torch 2.x and a modern transformers,
# so the environment has to be baked in here.
#
# Build on an amd64 machine or CI runner. Cross-building this on Apple Silicon works via
# `docker buildx --platform linux/amd64` but is slow; see cluster/README.md.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04 ships Python 3.10, which is new enough for our code and avoids the
# externally-managed-environment friction that 24.04 introduces.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /opt/app

# requirements-analysis.txt pulls in requirements.txt via `-r`, so this installs the base
# stack (torch, transformers, accelerate, datasets, textstat) plus scikit-learn,
# concept-erasure (LEACE) and markdown-it-py in one layer.
COPY requirements.txt requirements-analysis.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-analysis.txt

# Determined's harness runs inside the container; it must match the cluster master (0.35.0).
RUN python -m pip install determined==0.35.0

# Fail the build rather than the job if the stack is not actually importable.
RUN python -c "import torch, transformers, sklearn, concept_erasure, textstat; \
print('torch', torch.__version__, '| transformers', transformers.__version__)"

# NOTE: the code and data are NOT copied in. They live on PFSS and are bind-mounted at run
# time (see cluster/config.yaml), so a code change does not require an image rebuild.
