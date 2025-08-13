"""
Module for logging the requests and responses of the FastAPI app. Implements the concept of middleware so that
the logging is automatically done for every request and response instead of manually adding it to each endpoint.
"""

from fastapi import Request
from fastapi.responses import Response
from src.core import logger


async def log_middleware_request(request: Request, call_next):
    """
    Middleware to log requests
    Args:
        request: Request object
        call_next: Next middleware function
    Returns:
        Response object
    """
    log_dict = {
        "url": request.url,
        "method": request.method,
        "input": str(await request.body()),
    }
    logger.info(f"Request: {log_dict}")
    response = await call_next(request)
    return response


async def log_middleware_response(request: Request, call_next):
    """
    Middleware to log responses
    Args:
        request: Request object
        call_next: Next middleware function
    Returns:
        Response object
    """
    # Process request
    response = await call_next(request)

    # Log response
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    # Reconstruct response with body
    new_response = Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

    response_log = {
        "url": str(request.url),
        "method": request.method,
        "status_code": response.status_code,
        "response": response_body.decode() if response_body else None,
    }
    logger.info(f"Response: {response_log}")

    return new_response
