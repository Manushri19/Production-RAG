"""
Celery Application Configuration

Configures Celery for document processing workers.
Workers are stateless and process documents via HTTP services.
"""

from celery import Celery

from meridian.config import REDIS_URL

app = Celery(
    'meridian',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['meridian.workers.tasks'],
)

app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Reliability
    task_acks_late=True,           # Acknowledge after completion (retry on crash)
    worker_prefetch_multiplier=1,  # One task at a time per worker

    # Timeouts — large NTRS-class docs (300+ pages, 600+ VLM calls) can take
    # 8-15 min on first attempt under load. Generous limits eliminate the
    # retry storm we saw on big docs.
    task_time_limit=2100,          # Hard limit: 35 minutes
    task_soft_time_limit=1800,     # Soft limit: 30 minutes (raises exception)

    # Result expiration
    result_expires=3600,           # Results expire after 1 hour

    # Task routing (optional - for future multi-queue setup)
    task_default_queue='documents',

    # Retry settings
    task_default_retry_delay=60,   # 1 minute between retries
    task_max_retries=3,
)

# Optional: Task routing for different priorities
# app.conf.task_routes = {
#     'meridian.workers.tasks.process_document': {'queue': 'documents'},
#     'meridian.workers.tasks.process_document_urgent': {'queue': 'urgent'},
# }

if __name__ == '__main__':
    app.start()
