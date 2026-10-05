"""Synchronous work limits. Permits belong to running work, not waiting clients."""

import threading
import time
from contextlib import contextmanager


class ServiceError(Exception):
    def __init__(self, code, message, status_code=503, retryable=True):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class Deadline:
    def __init__(self, seconds, *, clock=time.monotonic):
        self.clock = clock
        self.end = clock() + seconds

    def remaining(self):
        return max(0.0, self.end - self.clock())

    def check(self):
        if self.remaining() <= 0:
            raise ServiceError("deadline", "The request time budget was exceeded.", 504)

    def timeout(self, maximum):
        self.check()
        return min(maximum, self.remaining())


class Admission:
    def __init__(self, limit):
        self._permits = threading.BoundedSemaphore(limit)

    @contextmanager
    def enter(self):
        if not self._permits.acquire(blocking=False):
            raise ServiceError("busy", "All chat slots are busy. Please retry shortly.")
        try:
            yield
        finally:
            self._permits.release()


class InferenceGate:
    def __init__(self, limit):
        self._permits = threading.BoundedSemaphore(limit)

    def run(self, function, *, deadline=None):
        if deadline:
            deadline.check()
        if not self._permits.acquire(blocking=False):
            raise ServiceError(
                "inference_busy", "Document processing is busy. Please retry shortly."
            )
        try:
            result = function()
            if deadline:
                deadline.check()
            return result
        finally:
            # Native CPU/GPU inference cannot be cancelled safely. Release only here.
            self._permits.release()
