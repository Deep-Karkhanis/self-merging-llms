FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN chown root:root /usr/lib

RUN apt update -y && apt install -y build-essential curl openssh-server openssh-client pdsh
RUN mkdir -p /var/run/sshd

RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

ENV SSH_PORT=4242
RUN cat /etc/ssh/ssh_config | grep -v Port > /etc/ssh/ssh_config.new && \
    echo "    Port ${SSH_PORT}" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN cat /etc/ssh/sshd_config > /tmp/sshd_config && \
    sed "0,/^#Port 22/s//Port ${SSH_PORT}/" /tmp/sshd_config > /etc/ssh/sshd_config

RUN pip install --upgrade pip wheel

RUN pip install \
        accelerate \
        aioprometheus \
        deepspeed \
        fastapi \
        fschat[model_worker,webui] \
        peft \
        protobuf==3.20.3 \
        ray \
        sentencepiece \
        transformers \
        trl \
        uvicorn

RUN pip install --no-deps stanford-stk

RUN mkdir /packages/

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/flash_attn-2.4.2-cp310-cp310-linux_aarch64.whl /packages
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/flash_attn-2.4.2-cp310-cp310-linux_x86_64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/vllm-0.2.7%2Bcu123-cp310-cp310-linux_aarch64.whl /packages
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/vllm-0.2.7%2Bcu123-cp310-cp310-linux_x86_64.whl /packages

ADD https://github.com/acollins3/triton/releases/download/triton-2.1.0-arm64/triton-2.1.0-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/xformers-0.0.24%2B6600003.d20240113-cp310-cp310-linux_aarch64.whl /packages
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/xformers-0.0.24%2B6600003.d20240113-cp310-cp310-linux_x86_64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/megablocks-0.5.1-cp310-cp310-linux_aarch64.whl /packages
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/megablocks-0.5.1-cp310-cp310-linux_x86_64.whl /packages

RUN pip install --no-deps --find-links /packages flash-attn==2.4.2

RUN pip install --no-deps --find-links /packages vllm==0.2.7

RUN pip install --no-deps --find-links /packages triton==2.1.0

RUN pip install --no-deps --find-links /packages xformers==0.0.24

RUN pip install --no-deps --find-links /packages megablocks==0.5.1

RUN rm -r /packages

RUN mkdir -p /root/.ssh && chmod -R go= /root/.ssh

COPY ./ssh-keys/* /root/.ssh/

RUN chmod 600 /root/.ssh/id_rsa || true

ENTRYPOINT ["/bin/bash"]
