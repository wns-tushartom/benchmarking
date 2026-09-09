"""Opt-in dashboard launcher; install beside serve_benchmark_dashboard.py.

Existing dashboard routes retain their original capabilities. Only the added
queue route is read-only. This launcher is not authentication or a sandbox.
"""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from queue_dashboard_handler import with_queue_status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5011)
    args = parser.parse_args(argv)
    if not args.state_dir.is_absolute() or '..' in args.state_dir.parts:
        parser.error('--state-dir must be an absolute server-owned path')
    if not 0 <= args.port <= 65535:
        parser.error('invalid port')
    return args


def create_server(args, base_handler):
    handler = with_queue_status(base_handler, args.state_dir)
    return ThreadingHTTPServer((args.host, args.port), handler)


def main(argv=None):
    args = parse_args(argv)
    # Deliberately defer baseline imports until explicit startup; --help has
    # no baseline import side effects. No existing module globals are patched.
    from serve_benchmark_dashboard import Handler
    server = create_server(args, Handler)
    try:
        print('Queue-enabled dashboard listening on', server.server_address, flush=True)
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
