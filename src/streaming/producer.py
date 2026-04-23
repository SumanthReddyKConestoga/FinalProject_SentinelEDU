"""StreamProducer — reads events.jsonl and pushes to a shared queue."""
import json
import queue
import threading
import time
from pathlib import Path

from src.config import SETTINGS, resolve_path
from src.utils.logging import get_logger

log = get_logger(__name__)

STREAM_FILE = resolve_path(SETTINGS["streaming"]["stream_file"])
RATE = SETTINGS["streaming"]["rate_events_per_sec"]


class StreamProducer(threading.Thread):
    """Reads events.jsonl line-by-line and puts them into a Queue."""

    def __init__(self, event_queue: queue.Queue, rate: float = RATE, loop: bool = True):
        super().__init__(daemon=True, name="StreamProducer")
        self.q = event_queue
        self.rate = rate
        self.loop = loop
        self._stop_event = threading.Event()
        self.events_produced = 0

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

    def run(self) -> None:
        delay = 1.0 / max(self.rate, 0.01)
        log.info(f"Producer started: file={STREAM_FILE}, rate={self.rate} evt/s")
        while not self._stop_event.is_set():
            try:
                self._emit_file(delay)
            except Exception as exc:
                log.error(f"Producer error: {exc}")
                time.sleep(2)
            if not self.loop:
                break
        log.info(f"Producer stopped. Events produced: {self.events_produced}")

    def _emit_file(self, delay: float) -> None:
        path = Path(STREAM_FILE)
        if not path.exists():
            log.warning(f"Stream file not found: {path}; waiting …")
            time.sleep(5)
            return
        with open(path, "r") as fh:
            for line in fh:
                if self._stop_event.is_set():
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    self.q.put(event)
                    self.events_produced += 1
                except json.JSONDecodeError:
                    log.warning(f"Skipping bad JSON line: {line[:80]}")
                time.sleep(delay)


# Module-level shared state
_shared_queue: queue.Queue = queue.Queue(maxsize=1000)
_producer: StreamProducer | None = None
_producer_lock = threading.Lock()


def get_queue() -> queue.Queue:
    return _shared_queue


def start_producer() -> StreamProducer:
    global _producer
    with _producer_lock:
        if _producer is not None and _producer.is_alive():
            log.info("Producer already running.")
            return _producer
        _shared_queue.queue.clear()
        _producer = StreamProducer(_shared_queue)
        _producer.start()
        return _producer


def stop_producer() -> None:
    global _producer
    with _producer_lock:
        if _producer and _producer.is_alive():
            _producer.stop()
            _producer.join(timeout=3)
            log.info("Producer stopped.")


def producer_status() -> dict:
    global _producer
    return {
        "running": _producer is not None and _producer.is_alive(),
        "events_produced": _producer.events_produced if _producer else 0,
    }
