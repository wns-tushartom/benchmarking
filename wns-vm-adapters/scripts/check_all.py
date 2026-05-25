from __future__ import annotations
import json, os, socket, urllib.error, urllib.request

def http_get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300, f"HTTP {resp.status} {resp.read(200).decode('utf-8', 'ignore')}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)

def http_post_json(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return 200 <= resp.status < 300, f"HTTP {resp.status}, count={body.get('count')}, dimensions={body.get('dimensions')}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.read().decode('utf-8', 'ignore')[:200]}"
    except Exception as exc:
        return False, str(exc)

def tcp_check(host, port, timeout=3):
    try:
        with socket.create_connection((host, port), timeout=timeout): return True, 'tcp ok'
    except Exception as exc: return False, str(exc)

def show(name, ok, msg):
    print(f"- {name}: {'OK' if ok else 'MISSING/CHECK'} - {msg}")
    return ok

def main():
    port = os.getenv('WNS_EMBEDDING_PORT', '5000')
    base = f'http://127.0.0.1:{port}'
    print('WNS VM adapter health check, port range 5000-5010 only')
    results=[]
    for name, fn in [
        ('embedding /health', lambda: http_get(base+'/health')),
        ('embedding /embed/jina', lambda: http_post_json(base+'/embed/jina', {'texts':['refund policy','flight change']})),
        ('embedding /embed/gte', lambda: http_post_json(base+'/embed/gte', {'texts':['refund policy','flight change']})),
        ('qdrant', lambda: http_get('http://127.0.0.1:5001/')),
        ('pgvector tcp', lambda: tcp_check('127.0.0.1', 5003)),
        ('weaviate /v1/meta', lambda: http_get('http://127.0.0.1:5004/v1/meta')),
    ]:
        ok,msg=fn(); results.append(show(name, ok, msg))
    print('\nBenchmark .env URLs:')
    print('JINA_EMBEDDING_URL=http://VM_HOST:5000/embed/jina')
    print('GTE_EMBEDDING_URL=http://VM_HOST:5000/embed/gte')
    print('QDRANT_URL=http://VM_HOST:5001')
    print('PGVECTOR_DSN=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark')
    print('WEAVIATE_URL=http://VM_HOST:5004')
    return 0 if results[0] else 1
if __name__ == '__main__': raise SystemExit(main())
