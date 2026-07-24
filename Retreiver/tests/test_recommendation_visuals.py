from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISUALS = ROOT / "web" / "recommendation-visuals.js"


def run_visual_probe(assertions: str) -> None:
    script = f"""
const V = require({str(VISUALS)!r});
{assertions}
"""
    proc = subprocess.run(
        ["node", "-e", script],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_category_colours_are_distinct_stable_and_filter_safe() -> None:
    run_visual_probe(
        """
const values = ['FAISS','PGVector','Qdrant','Weaviate','OpenSearch','CustomDB'];
const first = V.categoryColorMap(values);
const second = V.categoryColorMap([...values].reverse());
if (new Set(Object.values(first)).size !== 6) throw new Error(JSON.stringify(first));
if (JSON.stringify(first) !== JSON.stringify(second)) throw new Error('order-dependent colours');
const filtered = V.categoryColorMap(['Qdrant','FAISS'], first);
if (filtered.Qdrant !== first.Qdrant || filtered.FAISS !== first.FAISS) throw new Error('filter changed colours');
"""
    )


def test_canonical_rerankers_use_distinct_marker_shapes() -> None:
    run_visual_probe(
        """
if (V.rerankerShape('none') !== 'circle') throw new Error('none');
if (V.rerankerShape('No-reranker baseline') !== 'circle') throw new Error('baseline');
if (V.rerankerShape('bge-reranker-base') !== 'diamond') throw new Error('bge');
if (V.rerankerShape('Amazon Rerank v1') !== 'triangle') throw new Error('amazon');
if (V.rerankerShape('Qwen3:4B Rerank') !== 'square') throw new Error('fallback');
"""
    )


def test_family_keys_and_rank_views_preserve_canonical_order() -> None:
    run_visual_probe(
        """
const rows = Array.from({length:12}, (_,i)=>({
  combo_id:String(i), sheet:'c'+i, embedding:'e', store:'s', reranker:'none', x:i, y:12-i,
}));
if (V.familyKey(rows[0]) !== 'c0|e|s') throw new Error(V.familyKey(rows[0]));
if (V.applyQuickView(rows,'all').length !== 12) throw new Error('all');
if (V.applyQuickView(rows,'top10').map(r=>r.combo_id).join(',') !== '0,1,2,3,4,5,6,7,8,9') throw new Error('top10');
if (V.applyQuickView(rows,'bottom10').map(r=>r.combo_id).join(',') !== '2,3,4,5,6,7,8,9,10,11') throw new Error('bottom10');
"""
    )


def test_pareto_frontier_respects_metric_directions() -> None:
    run_visual_probe(
        """
const rows = [
  {id:'a',x:1,y:3},
  {id:'b',x:2,y:2},
  {id:'c',x:3,y:1},
  {id:'d',x:3,y:0},
];
const frontier = V.paretoRows(rows, r=>r.x, r=>r.y, 'lower', 'higher');
if (frontier.length !== 1 || frontier[0].id !== 'a') throw new Error(JSON.stringify(frontier));
const higherHigher = V.paretoRows(rows, r=>r.x, r=>r.y, 'higher', 'higher');
if (higherHigher.some(r=>r.id === 'd')) throw new Error(JSON.stringify(higherHigher));
"""
    )


def test_collision_offsets_spread_exact_overlaps_without_moving_singletons() -> None:
    run_visual_probe(
        """
const a={id:'a',x:1,y:2}, b={id:'b',x:1,y:2}, c={id:'c',x:4,y:5};
const offsets=V.collisionOffsets([a,b,c], r=>r.x, r=>r.y);
if (!offsets.get(a) || !offsets.get(b) || !offsets.get(c)) throw new Error('missing offsets');
if (offsets.get(a).dx === offsets.get(b).dx && offsets.get(a).dy === offsets.get(b).dy) throw new Error('overlap not spread');
if (offsets.get(c).dx !== 0 || offsets.get(c).dy !== 0) throw new Error('singleton moved');
"""
    )
