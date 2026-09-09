"""Read-only HTML queue page, using the same bounded public snapshot reader."""
from urllib.parse import urlsplit
from queue_status_reader import read_status
from queue_panel import render_panel


def handle_page_request(handler, state_dir):
    target = urlsplit(handler.path)
    if target.path != '/queue':
        return False
    if handler.command != 'GET':
        status, body = 405, 'Only GET is allowed.'
    elif handler.path != '/queue':
        status, body = 400, 'Query parameters are not supported.'
    else:
        status = 200
        body = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>Candidate queue — benchmark dashboard</title></head><body>'
                '<nav><a href="/">Back to benchmark dashboard</a></nav><main>'
                + render_panel(read_status(state_dir)) + '</main></body></html>')
    payload = body.encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'text/html; charset=utf-8')
    handler.send_header('Content-Length', str(len(payload)))
    handler.send_header('Cache-Control', 'no-store')
    handler.send_header('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
    handler.send_header('X-Content-Type-Options', 'nosniff')
    if status == 405:
        handler.send_header('Allow', 'GET')
    handler.end_headers()
    if handler.command != 'HEAD':
        handler.wfile.write(payload)
    return True
