"""Read-only route adapter for the existing dashboard request handler.

Caller supplies a server-owned absolute state directory, never a request value.
Caller must dispatch here for GET and mutation methods to enforce Allow: GET.
No server is started and no dashboard files are modified by importing this module.
"""
import json
from urllib.parse import urlsplit
from queue_status_reader import read_status


def handle_status_request(handler, state_dir):
    parsed = urlsplit(handler.path)
    if parsed.path != '/api/queue-status':
        return False
    if handler.command != 'GET':
        status, payload = 405, {'error': 'read_only'}
    elif parsed.query or parsed.fragment or parsed.scheme or parsed.netloc:
        status, payload = 400, {'error': 'query_not_supported'}
    else:
        status, payload = 200, read_status(state_dir)
    body = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Cache-Control', 'no-store, max-age=0')
    handler.send_header('X-Content-Type-Options', 'nosniff')
    if status == 405:
        handler.send_header('Allow', 'GET')
    handler.end_headers()
    if handler.command != 'HEAD':
        handler.wfile.write(body)
    return True
