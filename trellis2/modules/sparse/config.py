from typing import *
import platform
import sys

CONV = 'flex_gemm'
DEBUG = False
ATTN = 'flash_attn'

def __detect_defaults():
    """Auto-detect best backends for current platform."""
    global CONV, ATTN
    if platform.system() == 'Darwin':
        ATTN = 'sdpa'
        try:
            import flex_gemm
            CONV = 'flex_gemm'
        except ImportError:
            CONV = 'pytorch'
    elif not __has_cuda():
        CONV = 'pytorch'
        ATTN = 'sdpa'

def __has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def __from_env():
    import os

    global CONV
    global DEBUG
    global ATTN

    __detect_defaults()

    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm', 'pytorch']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']:
        ATTN = env_sparse_attn_backend

    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")


__from_env()


def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm', 'pytorch']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'sdpa']):
    global ATTN
    ATTN = backend
