# %% [markdown]
# # WNS VM Smoke Test, ports 5000-5010

# %%
import os, json, urllib.request
PORT = os.getenv('WNS_EMBEDDING_PORT', '5000')
BASE = f'http://127.0.0.1:{PORT}'

def get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.read().decode('utf-8')

def post_json(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())

# %%
get(BASE + '/health')

# %%
status, jina = post_json(BASE + '/embed/jina', {'texts': ['refund policy', 'flight change']})
status, jina['model'], jina['dimensions'], jina['count']

# %%
status, gte = post_json(BASE + '/embed/gte', {'texts': ['refund policy', 'flight change']})
status, gte['model'], gte['dimensions'], gte['count']
