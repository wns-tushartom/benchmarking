# Candidate queue source checkpoint

Status: NOT APPROVED FOR UNATTENDED PRODUCTION EXECUTION.
Full staging suite: 237 tests passed. Final inventory fix awaits independent closure.
No production selection contract, provider authorization or deployment is included.

This is an overlay, not the full repository. Preserve existing data, configs, environment,
results and VM-only remote_rerankers.py changes. Keep the spelling Retreiver.
Based on upstream branch feature/retrieval-reranker-candidates-vm at
2e8eb900f5d92f89e98a3a7208a90961d26e0725.

Before copying, review local changes to scripts/benchmark_cli.py,
benchmarking/core/runner.py and web/index.html. These three replace baseline files;
merge rather than overwrite if yours differ. All other files are queue additions.
Never replace the entire Retreiver directory or delete files absent from this overlay.

Queue Start/STOP remain CLI-only. Do not launch until production selection approval
and release review are complete. No candidate evidence is accepted automatically.
Git staging should use only the exact paths listed in queue-update-paths.md.
Check existing staged changes before staging; do not include unrelated staged files.
