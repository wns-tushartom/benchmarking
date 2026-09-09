"""Compose the queue route with an existing dashboard handler, no monkeypatch.

The startup caller supplies the state directory. This factory neither starts a
server nor installs into the dashboard. Existing endpoints retain their behavior.
"""
from pathlib import Path
from queue_status_http import handle_status_request
from queue_page_http import handle_page_request


def with_queue_status(base_handler, state_dir):
    state_dir = Path(state_dir)
    if not state_dir.is_absolute() or '..' in state_dir.parts:
        raise ValueError('absolute server-owned state directory required')

    class QueueDashboardHandler(base_handler):
        def _queue_or_existing(self, method):
            if handle_page_request(self, state_dir):
                return
            if handle_status_request(self, state_dir):
                return
            existing = getattr(super(), method, None)
            if existing is None:
                self.send_error(501, 'Unsupported method')
            else:
                existing()

        def do_GET(self):
            self._queue_or_existing('do_GET')

        def do_POST(self):
            self._queue_or_existing('do_POST')

        def do_HEAD(self):
            self._queue_or_existing('do_HEAD')

        def do_PUT(self):
            self._queue_or_existing('do_PUT')

        def do_PATCH(self):
            self._queue_or_existing('do_PATCH')

        def do_DELETE(self):
            self._queue_or_existing('do_DELETE')

        def do_OPTIONS(self):
            self._queue_or_existing('do_OPTIONS')

    return QueueDashboardHandler
