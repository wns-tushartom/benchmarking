from pathlib import Path
import hashlib,json,subprocess,sys,os
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release/FILES.json').read_text())
for name, expected in manifest.items():
 p=root/name
 assert p.is_file() and not p.is_symlink(), name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, name
print('All release file hashes verified',len(manifest),flush=True)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root)+os.pathsep+str(root/'tests'))
subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider']+['tests/test_portfolio_archived_status.py', 'tests/test_portfolio_next_safety.py', 'tests/test_dashboard_portfolio_control.py', 'tests/test_candidate_full_depth_admission.py', 'tests/test_meeting_400_dashboard_results.py', 'tests/test_recommendations_ui.py', 'tests/test_candidate_dashboard_frontend.py', 'tests/test_candidate_dashboard_options.py', 'tests/test_portfolio_batch_execution.py', 'tests/test_dashboard_metrics.py'],cwd=root,env=env,check=True)
for f in ['test_portfolio_release_ui.cjs','test_metrics_heatmap_integration.cjs','test_recommendation_visuals_lineage.cjs']:
 subprocess.run(['node','tests/'+f],cwd=root,check=True)
for p in (root/'web').glob('*.js'):
 subprocess.run(['node','--check',str(p)],check=True)
