import json
import tempfile
import unittest
from pathlib import Path
from queue_public_status import public_status
from queue_status_reader import read_status
from queue_panel import render_panel


class RunnerSnapshotProjectionTests(unittest.TestCase):
    def test_canonical_state_overrides_legacy_alias(self):
        for state in ('running', 'pending', 'completed', 'blocked', 'stopped_by_request'):
            with self.subTest(state=state):
                result = public_status({'queue_state':state,'state':'completed','jobs':[{'state':'completed'}], 'inventory':dict(master_configured=1, nonpaid_selected=1, paid_excluded=0, owned_by_jobs=1, pending_unassigned=0)})
                self.assertEqual(result['queue_state'],state)
        self.assertEqual(public_status({'queue_state':[], 'state':'completed','jobs':[]})['queue_state'],'unknown')

    def test_control_only_snapshot_preserves_state_not_job_counts(self):
        for state in ('stopped_by_request','blocked'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                raw = {'queue_state':state,'current_job':None,'inspection_verified':False,
                       'promotion_status':'not_accepted','failure':{'message':'PRIVATE_SENTINEL'}}
                (Path(tmp)/'queue_status.json').write_text(json.dumps(raw))
                result = read_status(Path(tmp))
                self.assertTrue(result['available'])
                self.assertEqual(result['queue_state'],state)
                self.assertIsNone(result['counts']['listed_jobs'])
                self.assertIsNone(result['counts']['recorded_completed_jobs'])
                self.assertIsNone(result['counts']['configured_combinations'])
                html = render_panel(result)
                self.assertIn('Recorded queue state: <strong>'+state.replace('_',' ')+'</strong>',html)
                self.assertNotIn('PRIVATE_SENTINEL',html)

    def test_contradictory_completed_jobs_are_unknown_in_reader_and_html(self):
        for key in ('queue_state', 'state'):
            for jobs in ([], [None], [{'state':'ready'}],
                         [{'state':'completed'}, {'state':'failed'}], [{}]):
                with self.subTest(key=key, jobs=jobs), tempfile.TemporaryDirectory() as tmp:
                    (Path(tmp)/'queue_status.json').write_text(json.dumps({key:'completed','jobs':jobs}))
                    result = read_status(Path(tmp))
                    self.assertEqual(result['queue_state'], 'unknown')
                    self.assertNotIn('Recorded queue state: <strong>completed</strong>', render_panel(result))

    def test_completed_requires_reconciled_fully_assigned_inventory(self):
        good = dict(master_configured=2, nonpaid_selected=2, paid_excluded=0,
                    owned_by_jobs=2, pending_unassigned=0)
        inventories = [None, {}, dict(good, pending_unassigned=1, owned_by_jobs=1),
                       dict(good, owned_by_jobs=1), dict(good, pending_unassigned=None)]
        for inventory in inventories:
            with self.subTest(inventory=inventory):
                result = public_status({'queue_state':'completed',
                    'jobs':[{'state':'completed'}], 'inventory':inventory})
                self.assertEqual(result['queue_state'], 'unknown')
        self.assertEqual(public_status({'queue_state':'completed',
            'jobs':[{'state':'completed'}], 'inventory':good})['queue_state'], 'completed')

    def test_malformed_or_success_without_job_evidence_unavailable(self):
        for raw in ({'queue_state':'completed'}, {'queue_state':'blocked','jobs':'bad'},
                    {'queue_state':[]}, {'state':'blocked'}):
            self.assertFalse(public_status(raw)['available'])
