"""Queue guards use temporary directories; never touch project experiment outputs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class UnifiedQueueTests(unittest.TestCase):
    def test_plan_is_exactly_the_authorized_main_experiments(self):
        from scripts.run_unified_baselines import make_tasks
        tasks = make_tasks()
        self.assertEqual(len(tasks), 36)  # 12 models x 3 datasets after HSLiNets removal
        self.assertEqual(len({t['key'] for t in tasks}), 36)
        for task in tasks:
            self.assertTrue(task['target'].endswith('/run_001'))
            self.assertNotIn('_fs10', task['config'])
            self.assertNotIn('_fs100', task['config'])

    def test_replacement_rejects_changed_results(self):
        from scripts.run_unified_baselines import fingerprint, replace_target
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'runs_fewshot/fs50/mft/yrd/run_001'
            target.mkdir(parents=True)
            (target / 'metrics.json').write_text('old')
            expected = fingerprint(target)
            (target / 'metrics.json').write_text('changed')
            with self.assertRaises(RuntimeError):
                replace_target(root, 'runs_fewshot/fs50/mft/yrd/run_001', expected, True)
            self.assertTrue(target.is_dir())

    def test_replacement_requires_flag_and_exact_whitelisted_path(self):
        from scripts.run_unified_baselines import fingerprint, replace_target
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = 'runs_fewshot/fs50/mft/yrd/run_001'
            target = root / relative
            target.mkdir(parents=True)
            (target / 'weights.pth').write_text('old')
            expected = fingerprint(target)
            with self.assertRaises(RuntimeError):
                replace_target(root, relative, expected, False)
            with self.assertRaises(ValueError):
                replace_target(root, 'runs_fewshot', expected, True)
            replace_target(root, relative, expected, True)
            self.assertFalse(target.exists())

    def test_memory_admission_accounts_for_reserved_and_external_usage(self):
        from scripts.run_unified_baselines import can_start
        self.assertTrue(can_start(4000, [6000, 6000], 24000, 2000, 15000))
        self.assertFalse(can_start(8000, [8000, 8000], 24000, 1000, 15000))
        self.assertFalse(can_start(4000, [], 24000, 1000, 3000))


if __name__ == '__main__':
    unittest.main()
