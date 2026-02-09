FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
 && mv bin/micromamba /usr/local/bin/micromamba

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n env -f /tmp/environment.yml \
 && micromamba clean -a -y

ENV PATH=/opt/conda/envs/env/bin:$PATH