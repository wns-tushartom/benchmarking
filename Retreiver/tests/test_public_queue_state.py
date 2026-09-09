import unittest
from queue_public_status import public_status

class PublicQueueStateTests(unittest.TestCase):
    def test_known_overall_states_survive_projection(self):
        for state in ('empty', 'completed', 'pending', 'running', 'blocked', 'waiting_external_producer', 'stopped_by_request'):
            with self.subTest(state=state):
                jobs = [{'state': 'completed'}] if state == 'completed' else []
                self.assertEqual(public_status({'state': state, 'jobs': jobs, 'inventory':dict(master_configured=1, nonpaid_selected=1, paid_excluded=0, owned_by_jobs=1, pending_unassigned=0)}).get('queue_state'), state)

    def test_unknown_private_and_malformed_states_are_not_exposed(self):
        for state in ('/private/log', 'token=secret', {}, [], None, True):
            with self.subTest(state=state):
                result = public_status({'state': state, 'jobs': []})
                self.assertEqual(result.get('queue_state'), 'unknown')
                self.assertFalse(result['receipt_revalidated'])

    def test_invalid_snapshot_does_not_advertise_completion(self):
        self.assertEqual(public_status({'state': 'completed'}).get('queue_state'), 'unknown')
