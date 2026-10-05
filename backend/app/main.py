import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .resources import Resources
from .runtime import Admission, Deadline, ServiceError
from .schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


def create_app(*, resources=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application):
        application.state.resources = resources or Resources(settings)
        try:
            yield
        finally:
            application.state.resources.close()

    application = FastAPI(title="Fraud Q&A Chatbot", lifespan=lifespan)
    admission = Admission(settings.max_concurrent_chats)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "request_id": request.state.request_id,
                }
            },
            headers={"Retry-After": "2"} if exc.retryable and exc.status_code == 503 else {},
        )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "Check question, history and document selection.",
                    "retryable": False,
                    "request_id": request.state.request_id,
                }
            },
        )

    @application.get("/live")
    async def live():
        return {"status": "ok"}

    @application.get("/health")
    def health():
        return application.state.resources.ready()

    @application.get("/ready")
    def ready():
        result = application.state.resources.ready()
        return JSONResponse(result, status_code=200 if result["status"] == "ok" else 503)

    @application.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request):
        from .agent.state_graph import run_agent

        deadline = Deadline(settings.request_timeout_seconds)
        with admission.enter():
            try:
                result = run_agent(payload, application.state.resources, deadline)
                result.request_id = request.state.request_id
                logger.info(
                    "chat_complete request_id=%s type=%s status=%s timings=%s",
                    result.request_id,
                    result.answer_type,
                    result.status,
                    result.timings,
                )
                return result
            except ServiceError:
                raise
            except Exception as exc:
                logger.error(
                    "chat_failed request_id=%s error_type=%s",
                    request.state.request_id,
                    type(exc).__name__,
                )
                raise ServiceError(
                    "internal_error",
                    "The request could not be completed.",
                    status_code=500,
                    retryable=False,
                ) from None

    return application


app = create_app()
