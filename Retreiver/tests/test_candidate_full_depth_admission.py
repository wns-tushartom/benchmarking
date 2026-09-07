import csv
import json
from test_meeting_400_dashboard_results import _repo_fixture, _write_verified_batch, _sha256, _write_json


def test_verified_short_run_is_archived_not_ranked(tmp_path):
    import scripts.serve_benchmark_dashboard as dashboard
    root, config, plan, plan_path = _repo_fixture(tmp_path)
    batch = _write_verified_batch(plan_path.parent, plan, 0)
    summary = batch / 'modular_summary.csv'
    with summary.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['query_count'] = '100'
    with summary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((batch / 'manifest.json').read_text())
    manifest['artifact_sha256']['modular_summary.csv'] = _sha256(summary)
    _write_json(batch / 'manifest.json', manifest)
    receipt = json.loads((batch / 'completion_receipt.json').read_text())
    receipt['artifact_sha256'] = manifest['artifact_sha256']
    receipt['manifest_sha256'] = _sha256(batch / 'manifest.json')
    _write_json(batch / 'completion_receipt.json', receipt)
    before = {p.name: _sha256(p) for p in batch.iterdir() if p.is_file()}
    result = dashboard.read_meeting_candidate_results(root=root, config_path=config)
    assert result['rows'] == []
    assert result['validated_row_count'] == 0
    assert result['archived_batch_count'] == 1
    assert result['batches'][0]['state'] == 'archived_query_depth'
    assert result['batches'][0]['required_query_count'] == 500
    assert before == {p.name: _sha256(p) for p in batch.iterdir() if p.is_file()}
