"""Async utilities for managing async operations."""

import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

logger = logging.getLogger("temporal-graphrag.utils.async_utils")

P = ParamSpec('P')
T = TypeVar('T')


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get an event loop, creating one if necessary.
    
    In the main thread, gets the existing event loop.
    In sub-threads, creates a new event loop.
    
    Returns:
        asyncio.AbstractEventLoop: The event loop to use
        
    Example:
        >>> loop = always_get_an_event_loop()
        >>> # Use loop for async operations
    """
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def limit_async_func_call(max_size: int, waiting_time: float = 0.0001) -> Callable:
    """
    Decorator to limit the maximum number of concurrent async calls.
    
    Uses a simple counter-based approach instead of asyncio.Semaphore
    to avoid issues with nest-asyncio.
    
    Args:
        max_size: Maximum number of concurrent async calls allowed
        waiting_time: Time to wait (in seconds) between checks when limit is reached
        
    Returns:
        Decorator function
        
    Example:
        >>> @limit_async_func_call(max_size=10)
        ... async def my_async_function(arg):
        ...     # This function will only allow 10 concurrent calls
        ...     pass
    """
    def final_decorator(func: Callable[P, T]) -> Callable[P, T]:
        """Not using async.Semaphore to avoid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waiting_time)
            __current_size += 1
            try:
                result = await func(*args, **kwargs)
            finally:
                __current_size -= 1
            return result

        return wait_func  # type: ignore

    return final_decorator


__all__ = ["always_get_an_event_loop", "limit_async_func_call"]

