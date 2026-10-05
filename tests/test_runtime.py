import threading
import time

import pytest


def test_deadline_bounds_timeout_and_rejects_expiry():
    from backend.app.runtime import Deadline, ServiceError

    now = [10.0]
    deadline = Deadline(2, clock=lambda: now[0])
    assert deadline.timeout(30) == 2
    now[0] = 12
    with pytest.raises(ServiceError, match="budget") as exc:
        deadline.check()
    assert exc.value.status_code == 504


def test_permit_is_held_until_inference_actually_returns():
    from backend.app.runtime import Deadline, InferenceGate, ServiceError

    gate = InferenceGate(1)
    started, finish = threading.Event(), threading.Event()
    errors = []

    def work():
        started.set()
        finish.wait(2)

    def worker():
        try:
            gate.run(work, deadline=Deadline(0.01))
        except ServiceError as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(1)
    time.sleep(0.02)
    with pytest.raises(ServiceError) as exc:
        gate.run(lambda: None, deadline=Deadline(0.01))
    assert exc.value.status_code == 503
    finish.set()
    thread.join(2)
    assert errors[0].status_code == 504
    assert gate.run(lambda: "released", deadline=Deadline(1)) == "released"


def test_admission_saturation_does_not_queue():
    from backend.app.runtime import Admission, ServiceError

    gate = Admission(1)
    with gate.enter():
        with pytest.raises(ServiceError) as exc:
            with gate.enter():
                pass
        assert exc.value.code == "busy"
    with gate.enter():
        pass
