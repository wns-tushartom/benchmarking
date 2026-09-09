#!/usr/bin/env python3
"""Fail-closed sequential controller for immutable WNS portfolio batches."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class ContractError(RuntimeError):
    """The target is not the contracted VM/repository state."""


class LaunchError(RuntimeError):
    """An approved benchmark subprocess failed or produced no valid receipt."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _default_command_runner(command: Sequence[str], **kwargs: Any) -> CommandResult:
    from queue_child import run_child
    result = run_child(command, **kwargs)
    return CommandResult(result['returncode'], '',
                         'child timed out' if result['timed_out'] else '')


def _default_external_probe(session: str) -> bool:
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", '=' + session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ContractError('external producer probe unavailable; launch blocked') from None
    if result.returncode not in (0, 1):
        raise ContractError('external producer probe failed; launch blocked')
    return result.returncode == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
        path.chmod(0o600)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _lookup(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ContractError(f"config assertion path is missing: {dotted}")
        value = value[part]
    return value


class QueueController:
    def __init__(
        self,
        repo_root: Path,
        spec_path: Path,
        *,
        state_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
        command_runner: Callable[..., CommandResult] = _default_command_runner,
        external_probe: Callable[[str], bool] = _default_external_probe,
        receipt_verifier: Callable[[Path, Any, Any, str], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = 30.0,
    ) -> None:
        self.repo_root = Path(os.path.abspath(repo_root))
        self.spec_path = Path(os.path.abspath(spec_path))
        self.state_dir = Path(state_dir or self.spec_path.parent / "state")
        self.env = dict(os.environ if env is None else env)
        self.command_runner = command_runner
        self.external_probe = external_probe
        self.receipt_verifier = receipt_verifier
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self._api_loaded = False

    def _load_api(self) -> None:
        if self._api_loaded:
            return
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        try:
            from benchmarking.core.config import generate_matrix_catalog, load_benchmark_config
            from benchmarking.core.portfolio import build_portfolio_plan
            from scripts.publish_portfolio_results import verify_completed_batch
        except Exception as exc:
            raise ContractError(f"repository APIs are unavailable: {exc}") from exc
        self.generate_matrix_catalog = generate_matrix_catalog
        self.load_benchmark_config = load_benchmark_config
        self.build_portfolio_plan = build_portfolio_plan
        self.verify_completed_batch = verify_completed_batch
        self._api_loaded = True

    def _load_spec(self) -> dict[str, Any]:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ContractError('duplicate queue spec key')
                result[key] = value
            return result

        if self.spec_path.resolve() != self.spec_path or not self.spec_path.is_file():
            raise ContractError('queue spec must be a canonical regular file')
        try:
            payload = json.loads(self.spec_path.read_text(encoding="utf-8"),
                                 object_pairs_hook=unique_object)
        except Exception:
            raise ContractError('queue spec is unreadable or ambiguous') from None
        if (not isinstance(payload, dict) or type(payload.get('schema_version')) is not int
                or payload['schema_version'] != 1):
            raise ContractError("queue spec schema_version must be 1")
        if not isinstance(payload.get("contract"), dict) or not isinstance(payload.get("jobs"), list):
            raise ContractError("queue spec must contain contract and jobs")
        for job in payload['jobs']:
            if not isinstance(job, dict):
                raise ContractError('queue job must be an object')
            if 'external_owner' in job:
                owner = job['external_owner']
                session = owner.get('tmux_session') if isinstance(owner, dict) else None
                if (not isinstance(session, str) or not session.strip()
                        or session != session.strip()
                        or any(ord(char) < 32 or ord(char) == 127 for char in session)):
                    raise ContractError('declared external owner requires an exact nonempty tmux session')
        return payload

    def _stop_requested(self) -> bool:
        # Any directory entry means STOP, including dangling symlinks.
        return os.path.lexists(self.state_dir / 'STOP')

    def _safe_repo_path(self, relative: str, *, must_exist: bool = True) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ContractError(f"repository path must be nonempty and relative: {relative!r}")
        path = self.repo_root / relative
        try:
            path.relative_to(self.repo_root)
        except ValueError as exc:
            raise ContractError(f"repository path escapes root: {relative}") from exc
        current = self.repo_root
        for component in Path(relative).parts:
            if component in {".", ".."}:
                raise ContractError(f"repository path contains unsafe component: {relative}")
            current = current / component
            if current.is_symlink():
                raise ContractError(f"repository path must not contain symlinks: {relative}")
        if must_exist and not path.is_file():
            raise ContractError(f"required repository file is missing: {relative}")
        return path

    def _validate_contract(self, spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self.repo_root.is_symlink() or not self.repo_root.is_dir() or self.repo_root.resolve() != self.repo_root:
            raise ContractError("repo root is missing, noncanonical, or a symlink")
        contract = spec["contract"]
        source_hashes = contract.get('required_source_sha256', {})
        if not isinstance(source_hashes, dict):
            raise ContractError('source hash contract must be a mapping')
        for relative, expected in source_hashes.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(char not in '0123456789abcdef' for char in expected)):
                raise ContractError('invalid approved source hash')
            if _sha256(self._safe_repo_path(relative)) != expected:
                raise ContractError('approved source hash mismatch')
        for relative in contract.get("required_source_files", []):
            self._safe_repo_path(relative)
        for relative, expected in contract.get("required_data_sha256", {}).items():
            path = self._safe_repo_path(relative)
            actual = _sha256(path)
            if actual != expected:
                raise ContractError(f"required data hash mismatch for {relative}: {actual}")

        self._load_api()
        master_path = self._safe_repo_path(contract.get("master_config", ""))
        master = self.load_benchmark_config(master_path)
        catalog = self.generate_matrix_catalog(master)
        configured = catalog.get("configured")
        if not isinstance(configured, list) or not configured:
            raise ContractError("master portfolio generated no configured combinations")
        return master, configured

    @staticmethod
    def _paid(row: Mapping[str, Any], master: Mapping[str, Any], paid_adapters: set[str]) -> bool:
        techniques = master.get("techniques", {})
        for axis, key in (("embeddings", "embedding"), ("rerankers", "reranker")):
            item = techniques.get(axis, {}).get(row.get(key), {})
            if item.get("license") == "commercial" or item.get("adapter") in paid_adapters:
                return True
        return False

    def _default_receipt_check(self, portfolio_root: Path, plan: Any, batch: Any, lane: str) -> bool:
        try:
            from queue_completion import verify_completion
            spec = self._load_spec()
            matches = [job for job in spec['jobs']
                       if job.get('expected_portfolio_id') == plan.portfolio_id
                       and job.get('batch_id') == batch.batch_id and job.get('lane') == lane]
            if len(matches) != 1:
                return False
            config = self.load_benchmark_config(self._safe_repo_path(matches[0]['config']))
            from benchmarking.core.config import config_hash
            if matches[0].get('approved_config_hash') != config_hash(config):
                return False
            verify_completion(self.repo_root, portfolio_root, batch.batch_id, config,
                              spec['contract'].get('required_data_sha256', {}),
                              metrics=spec['contract'].get('required_metrics', []))
        except Exception:
            return False
        return True

    def _job_context(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        required = ("name", "config", "lane", "expected_portfolio_id", "batch_id")
        if any(not isinstance(raw.get(key), str) or not raw[key] for key in required):
            raise ContractError("every job requires nonempty name/config/lane/portfolio/batch identifiers")
        lane = raw["lane"]
        if len(Path(lane).parts) != 1 or lane in {".", ".."}:
            raise ContractError(f"unsafe output lane for job {raw['name']}")
        config_path = self._safe_repo_path(raw["config"])
        config = self.load_benchmark_config(config_path)
        configured_lane = str(config.get("experiment", {}).get("output_lane", ""))
        if configured_lane != lane:
            raise ContractError(f"lane mismatch for job {raw['name']}: {configured_lane!r}")
        for dotted, expected in raw.get("config_assertions", {}).items():
            actual = _lookup(config, dotted)
            if actual != expected:
                raise ContractError(
                    f"config assertion/model mismatch for job {raw['name']} at {dotted}: {actual!r}"
                )
        plan = self.build_portfolio_plan(config)
        if plan.portfolio_id != raw["expected_portfolio_id"]:
            raise ContractError(f"portfolio mismatch for job {raw['name']}: {plan.portfolio_id}")
        batch = next((candidate for candidate in plan.batches if candidate.batch_id == raw["batch_id"]), None)
        if batch is None:
            raise ContractError(f"batch mismatch for job {raw['name']}: {raw['batch_id']}")
        portfolio_root = self._safe_repo_path(
            str(Path('data') / 'modular_runs' / lane / plan.portfolio_id), must_exist=False)
        batch_dir = self._safe_repo_path(
            str(portfolio_root.relative_to(self.repo_root) / batch.batch_id), must_exist=False)
        return {
            "raw": raw,
            "config_path": config_path,
            "config": config,
            "plan": plan,
            "batch": batch,
            "portfolio_root": portfolio_root,
            "batch_dir": batch_dir,
            "plan_path": portfolio_root / "portfolio_plan.json",
        }

    def _readiness(self, raw: Mapping[str, Any]) -> tuple[bool, str]:
        readiness = raw.get("readiness")
        if not isinstance(readiness, Mapping) or readiness.get("approved") is not True:
            reason = readiness.get("reason") if isinstance(readiness, Mapping) else None
            return False, str(reason or "not readiness-approved")
        mismatches = []
        for key, expected in readiness.get("required_env_exact", {}).items():
            if self.env.get(key) != expected:
                mismatches.append(key)
        if mismatches:
            return False, "environment contract mismatch: " + ", ".join(sorted(mismatches))
        return True, "readiness approved"

    def inspect(self) -> dict[str, Any]:
        spec = self._load_spec()
        master, configured = self._validate_contract(spec)
        paid_adapters = set(spec["contract"].get("excluded_paid_adapters", ["openai", "amazon_bedrock"]))
        paid_ids = {row["combination_id"] for row in configured if self._paid(row, master, paid_adapters)}
        nonpaid_ids = {row["combination_id"] for row in configured} - paid_ids

        active_sessions = []
        for raw in spec['jobs']:
            owner = raw.get('external_owner', {})
            session = owner.get('tmux_session') if isinstance(owner, Mapping) else None
            if isinstance(session, str) and session and session not in active_sessions:
                if self.external_probe(session):
                    active_sessions.append(session)
        contexts = [self._job_context(raw) for raw in spec["jobs"]]
        owned: set[str] = set()
        jobs: list[dict[str, Any]] = []
        verifier = self.receipt_verifier or self._default_receipt_check
        for context in contexts:
            raw, plan, batch = context["raw"], context["plan"], context["batch"]
            batch_ids = set(batch.combination_ids)
            outside = batch_ids - nonpaid_ids
            if outside:
                raise ContractError(f"job {raw['name']} contains paid or non-master combinations")
            duplicate = owned & batch_ids
            if duplicate:
                raise ContractError(f"job {raw['name']} overlaps another queue job")
            owned.update(batch_ids)

            completed = False
            if active_sessions:
                # Wait globally before reading any batch completion artifacts.
                state, reason = 'waiting_external_producer', 'configured external producer active; global wait'
            elif context["batch_dir"].exists() or context["batch_dir"].is_symlink():
                if context["batch_dir"].is_symlink():
                    raise ContractError(f"batch output is a symlink for job {raw['name']}")
                completed = bool(verifier(context["portfolio_root"], plan, batch, raw["lane"]))
                state = "completed" if completed else "pending_partial_output"
                reason = "valid immutable completion receipt" if completed else "existing output lacks a valid completion receipt; preserved and blocked"
            else:
                ready, reason = self._readiness(raw)
                state = "ready" if ready else "pending_readiness"
            jobs.append({
                "name": raw["name"],
                "state": state,
                "reason": reason,
                "config": raw["config"],
                "lane": raw["lane"],
                "portfolio_id": plan.portfolio_id,
                "batch_id": batch.batch_id,
                "embedding": batch.embedding,
                "reranker_group": batch.reranker_group,
                "rerankers": list(batch.rerankers),
                "combination_count": len(batch.combination_ids),
                "output_dir": str(context["batch_dir"]),
            })

        pending = sorted(nonpaid_ids - owned)
        from queue_method_inventory import method_inventory
        methods = method_inventory(configured, paid_ids, owned)
        return {
            "method_inventory": methods,
            "schema_version": 1,
            "repo_root": str(self.repo_root),
            "master_config": spec["contract"]["master_config"],
            "inventory": {
                "master_configured": len(configured),
                "paid_excluded": len(paid_ids),
                "nonpaid_selected": len(nonpaid_ids),
                "owned_by_jobs": len(owned),
                "pending_unassigned": len(pending),
            },
            "paid_exclusions": {
                "state": "excluded_pending_authorization",
                "combination_count": len(paid_ids),
                "combination_ids": sorted(paid_ids),
            },
            "pending_unassigned": {
                "state": "pending_unsupported_or_unavailable",
                "reason": "no readiness-approved isolated immutable job config is declared",
                "combination_count": len(pending),
                "combination_ids": pending,
            },
            "jobs": jobs,
        }

    def _write_status(self, status: Mapping[str, Any]) -> None:
        _atomic_json(self.state_dir / "queue_status.json", status)

    def _ensure_plan(self, context: Mapping[str, Any]) -> Path:
        """Write or verify the canonical plan derived by the repository API."""
        plan_path = context["plan_path"]
        self._safe_repo_path(str(plan_path.relative_to(self.repo_root)), must_exist=False)
        portfolio_root = context["portfolio_root"]
        for path in (portfolio_root.parent, portfolio_root, plan_path):
            if path.is_symlink():
                raise ContractError("canonical portfolio plan path contains a symlink")
        if plan_path.exists():
            try:
                current = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ContractError("existing immutable portfolio plan is unreadable") from exc
            if current != context["plan"].to_dict():
                raise ContractError("existing immutable portfolio plan differs from repository-derived plan")
            return plan_path
        _atomic_json(plan_path, context["plan"].to_dict())
        return plan_path

    def _launch(self, job: Mapping[str, Any]) -> None:
        spec = self._load_spec()
        raw = next(item for item in spec["jobs"] if item["name"] == job["name"])
        context = self._job_context(raw)
        from benchmarking.core.config import config_hash
        from queue_preflight_bridge import run_preflight
        approved_hash = raw.get('approved_config_hash')
        if not approved_hash or approved_hash != config_hash(context['config']):
            raise ContractError('independently approved configuration hash required')
        payload = {'schema_version': 1, 'root': str(self.repo_root),
            'config': context['config'], 'approved_plan': context['plan'].to_dict(),
            'batch_id': context['batch'].batch_id, 'approved_config_hash': approved_hash,
            'approved_hashes': spec['contract'].get('required_data_sha256', {}),
            'allow_openai': spec['contract'].get('allow_openai', False)}
        try:
            evidence = run_preflight(payload, state_dir=self.state_dir, env=self.env,
                timeout=spec['contract'].get('preflight_timeout_seconds', 120),
                pass_fds=((self._repository_lock_fd,) if getattr(self, '_repository_lock_fd', None) is not None else ()))
        except Exception:
            raise LaunchError('batch preflight failed; benchmark not launched') from None
        if evidence.get('preflight_verified') is not True:
            raise LaunchError('batch preflight did not verify')
        # Recheck state after potentially slow provider inference.
        if self._stop_requested() or self._load_spec() != spec:
            raise LaunchError('queue stopped or selection changed during preflight')
        current = self.inspect()
        if any(item['state'] == 'waiting_external_producer' for item in current['jobs']):
            raise LaunchError('external producer detected after preflight')
        selected = next(item for item in current['jobs'] if item['name'] == job['name'])
        if selected['state'] != 'ready':
            raise LaunchError('batch no longer eligible after preflight')
        context = self._job_context(raw)
        if config_hash(context['config']) != approved_hash:
            raise LaunchError('configuration changed during preflight')
        corpus = evidence.get('corpus')
        if not isinstance(corpus, dict) or corpus.get('corpus_verified') is not True:
            raise LaunchError('positive corpus admission evidence required')
        if 'corpus_schema_and_depth' in evidence.get('pending_gates', []):
            raise LaunchError('corpus schema/depth admission is still pending')
        plan_path = self._ensure_plan(context)
        command = [
            sys.executable,
            str(self.repo_root / "scripts" / "benchmark_cli.py"),
            "portfolio-run-batch",
            str(context["config_path"]),
            "--plan",
            str(plan_path),
            "--batch-id",
            context["batch"].batch_id,
        ]
        import uuid
        log_path = self.state_dir / ('child-' + uuid.uuid4().hex + '.log')
        lock_fd = getattr(self, '_repository_lock_fd', None)
        inheritance = {'pass_fds': (lock_fd,)} if lock_fd is not None else {}
        # Revalidate approved file hashes after plan creation as well. A prior
        # preflight receipt cannot authorize bytes that changed afterward.
        try:
            self._validate_reviewed_sources(spec)
            self._validate_contract(spec)
            final_context = self._job_context(raw)
            if (config_hash(final_context['config']) != approved_hash
                    or final_context['plan'].to_dict() != context['plan'].to_dict()
                    or final_context['batch_dir'] != context['batch_dir']):
                raise ValueError('approved configuration or batch changed')
        except Exception:
            raise LaunchError('approved input contract changed before launch') from None
        # Probe configured owners again after plan creation, before the final
        # output check. Probe failure is a launch blocker, never permission.
        try:
            sessions = {item['external_owner']['tmux_session'] for item in spec['jobs']
                        if 'external_owner' in item}
            if any(self.external_probe(session) for session in sorted(sessions)):
                raise LaunchError('external producer detected before launch')
        except Exception:
            raise LaunchError('external producer ownership not clear before launch') from None
        # Plan creation can take place after the earlier eligibility check.
        # Fail closed if output appeared meanwhile; never merge partial work.
        # This is a launch-adjacent guard, not exclusion of uncooperative writers.
        self._safe_repo_path(str(context['batch_dir'].relative_to(self.repo_root)), must_exist=False)
        if os.path.lexists(context['batch_dir']):
            raise LaunchError('batch output appeared before launch; preserved without launching')
        if self._stop_requested() or self._load_spec() != spec:
            raise LaunchError('queue stopped or selection changed before launch')
        result = self.command_runner(command, cwd=self.repo_root, env=self.env,
            log_path=log_path, timeout=spec['contract'].get('batch_timeout_seconds', 86400),
            **inheritance)
        if result.returncode != 0:
            raise LaunchError(
                f"job {job['name']} exited {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
            )

    def _validate_execution_contract(self):
        spec = self._load_spec()
        hashes = spec.get('contract', {}).get('required_source_sha256')
        required = {'scripts/benchmark_cli.py', 'benchmarking/core/runner.py'}
        if not isinstance(hashes, dict) or not required.issubset(hashes):
            raise ContractError('run requires approved writer and CLI source hashes')
        self._validate_reviewed_sources(spec)
        self._validate_contract(spec)

    def _validate_reviewed_sources(self, spec):
        # This installed package resource, not the selectable spec, defines
        # reviewed writer identity. It does not confer deployment approval.
        try:
            pins = json.loads(Path(__file__).with_name('queue_writer_source_contract.json').read_text())
            required = {'scripts/benchmark_cli.py', 'benchmarking/core/runner.py'}
            reviewed = pins['after_sha256']
            selected = spec['contract']['required_source_sha256']
            if set(reviewed) != required or any(selected.get(p) != reviewed[p] for p in required):
                raise ValueError('unreviewed source')
        except Exception:
            raise ContractError('execution requires pinned reviewed writer source hashes') from None

    def run(self, *, wait: bool = True) -> dict[str, Any]:
        # All controller state directories share one lock for this repository.
        # Never unlink the lock: replacing its inode would defeat exclusion.
        import stat
        if self.repo_root.resolve() != self.repo_root or not self.repo_root.is_dir():
            raise ContractError('repository lock requires a canonical directory')
        try:
            descriptor = os.open(self.repo_root / '.candidate-queue.lock',
                                 os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        except OSError:
            raise ContractError('repository lock cannot be safely opened') from None
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ContractError('repository lock must be a regular, unaliased file')
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ContractError('another controller holds the repository queue lock') from None
            self._repository_lock_fd = descriptor
            return self._run_locked(wait=wait)
        finally:
            self._repository_lock_fd = None
            os.close(descriptor)

    def _run_locked(self, *, wait: bool = True) -> dict[str, Any]:
        import stat
        self.state_dir = Path(os.path.abspath(self.state_dir))
        if self.state_dir.resolve() != self.state_dir:
            raise ContractError('queue state path must not contain symlinks')
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.state_dir / "queue.lock"
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        except OSError:
            raise ContractError('queue state lock cannot be safely opened') from None
        with os.fdopen(descriptor, 'a+', encoding='utf-8') as lock:
            info = os.fstat(lock.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ContractError('queue state lock must be a regular, unaliased file')
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ContractError("another queue controller holds the queue lock") from exc
            while True:
                if self._stop_requested():
                    status = {'queue_state': 'stopped_by_request', 'current_job': None,
                              'inspection_verified': False, 'promotion_status': 'not_accepted'}
                    self._write_status(status)
                    return status
                try:
                    status = self.inspect()
                except Exception as exc:
                    self._write_status({
                        'queue_state': 'blocked', 'current_job': None,
                        'inspection_verified': False,
                        'promotion_status': 'not_accepted',
                        'failure': {'stage': 'inspection', 'error_type': type(exc).__name__,
                                    'message': 'inspection failed; no launch; completion unknown'},
                    })
                    raise ContractError('queue inspection failed; inspect private status') from None
                if self._stop_requested():
                    status["queue_state"] = "stopped_by_request"
                    self._write_status(status)
                    return status
                self._write_status(status)
                external = [job for job in status["jobs"] if job["state"] == "waiting_external_producer"]
                if external:
                    status.update(queue_state='waiting_external_producer', current_job=None)
                    self._write_status(status)
                    if not wait:
                        return status
                    self.sleep(self.poll_seconds)
                    continue
                ready = next((job for job in status["jobs"] if job["state"] == "ready"), None)
                if ready is None:
                    jobs = status['jobs']
                    unassigned = status.get('pending_unassigned', {})
                    has_unassigned = bool(unassigned.get('combination_count') or
                                          unassigned.get('combination_ids'))
                    terminal_state = ('pending' if has_unassigned else
                                      'empty' if not jobs else
                                      'completed' if all(job['state'] == 'completed' for job in jobs)
                                      else 'pending')
                    status.update(queue_state=terminal_state, current_job=None)
                    self._write_status(status)
                    return status
                self._validate_execution_contract()
                status.update(queue_state='running', current_job=ready['name'])
                self._write_status(status)
                stage = 'launch'
                try:
                    self._launch(ready)
                    stage = 'completion_verification'
                    after = self.inspect()
                    launched = next(job for job in after['jobs'] if job['name'] == ready['name'])
                    if launched['state'] != 'completed':
                        raise LaunchError('child returned without verified completion; output preserved')
                    after.update(queue_state='running', current_job=None)
                    self._write_status(after)
                except Exception as exc:
                    status.update(queue_state='blocked', current_job=ready['name'],
                        failure={'stage': stage, 'error_type': type(exc).__name__,
                                 'message': 'queue stopped; outputs preserved; completion not assumed'})
                    self._write_status(status)
                    raise LaunchError('queue blocked; inspect private queue status and logs') from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"), nargs="?", default="run")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True,
                        help="path to the approved queue selection contract (no implicit default)")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-wait", action="store_true", help="report an external producer instead of polling")
    args = parser.parse_args(argv)
    try:
        controller = QueueController(
            args.repo_root,
            args.spec,
            state_dir=args.state_dir,
            poll_seconds=args.poll_seconds,
        )
        if args.command == "run":
            spec = controller._load_spec()
            hashes = spec.get('contract', {}).get('required_source_sha256')
            required = {'scripts/benchmark_cli.py', 'benchmarking/core/runner.py'}
            if not isinstance(hashes, dict) or not required.issubset(hashes):
                raise ContractError('run requires approved writer and CLI source hashes')
            controller._validate_contract(spec)
        status = controller.inspect() if args.command == "status" else controller.run(wait=not args.no_wait)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    except (ContractError, LaunchError) as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
