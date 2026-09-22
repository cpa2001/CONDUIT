from __future__ import annotations

import base64
import mimetypes
import os


def _local_file_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    with open(path, 'rb') as file:
        encoded = base64.b64encode(file.read()).decode('utf-8')
    return f'data:{mime};base64,{encoded}'


def ensure_image_url(value: str) -> str:
    if value.startswith(('http://', 'https://', 'data:image')):
        return value
    if os.path.exists(value):
        return _local_file_to_data_url(value)
    return value


def ensure_video_url(value: str) -> str:
    if value.startswith(('http://', 'https://', 'data:video')):
        return value
    if os.path.exists(value):
        return _local_file_to_data_url(value)
    return value
