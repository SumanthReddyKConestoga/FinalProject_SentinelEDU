"""Streaming control routes."""
from fastapi import APIRouter

from src.streaming.producer import (
    start_producer, stop_producer, producer_status, get_queue,
)
from src.streaming.consumer import (
    start_consumer, stop_consumer, consumer_status,
)

router = APIRouter(prefix="/streaming", tags=["streaming"])


@router.post("/start")
def start_stream():
    q = get_queue()
    start_producer()
    start_consumer(q)
    return {"status": "started"}


@router.post("/stop")
def stop_stream():
    stop_producer()
    stop_consumer()
    return {"status": "stopped"}


@router.get("/status")
def stream_status():
    p = producer_status()
    c = consumer_status()
    q = get_queue()
    return {
        "running": p["running"] or c["running"],
        "producer": p,
        "consumer": c,
        "queue_depth": q.qsize(),
    }
