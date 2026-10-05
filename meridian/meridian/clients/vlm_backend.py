"""
VLM backend selection.

The VLM step reads cropped page regions -- tables, formulas, figures -- and
turns them into markdown, LaTeX and descriptions. It can run against either a
locally hosted vLLM server or the AWS Bedrock Converse API. Both backends
expose the same ``process_vlm_tasks_async(tables, figures, formulas, context)``
coroutine and return the same schema, so the rest of the pipeline neither
knows nor cares which one is in use.

Select with the ``VLM_BACKEND`` environment variable:

    VLM_BACKEND=vllm      # default -- local GPU, fastest, no per-token cost
    VLM_BACKEND=bedrock   # no local vision model needed; pay per token

Reach for ``bedrock`` when you do not have a GPU big enough to hold a vision
model alongside Docling. Docling itself still runs locally and is much faster
on a GPU, but ``DOCLING_ACCELERATOR_DEVICE=cpu`` works if you have none --
that combination is the lowest-hardware way to run the pipeline.

Bedrock needs ``boto3`` (``pip install 'meridian[bedrock]'``) and standard AWS
credentials from the environment, shared config, or an instance role.
"""

import logging
import os

logger = logging.getLogger(__name__)

VLM_BACKEND = os.getenv("VLM_BACKEND", "vllm").strip().lower() or "vllm"

_SUPPORTED = ("vllm", "bedrock")


def get_process_vlm_tasks_async():
    """Return the ``process_vlm_tasks_async`` coroutine for the configured backend.

    Imports are deferred so that a vLLM deployment never needs boto3 installed,
    and a Bedrock deployment never needs a local vision model configured.
    """
    if VLM_BACKEND == "bedrock":
        from meridian.clients.vlm_client_bedrock import process_vlm_tasks_async

        logger.info("VLM backend: bedrock")
        return process_vlm_tasks_async

    if VLM_BACKEND == "vllm":
        from meridian.clients.vlm_client_async import process_vlm_tasks_async

        logger.info("VLM backend: vllm")
        return process_vlm_tasks_async

    raise ValueError(
        f"Unknown VLM_BACKEND {VLM_BACKEND!r}. Expected one of: {', '.join(_SUPPORTED)}."
    )
