"""CPU synthetic checks; never load project datasets or train an experiment."""
from __future__ import annotations

import importlib
import importlib.util
import io
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable in this Python environment")
class HsiAdapterModelsTests(unittest.TestCase):
    def test_forward_backward_parameters_and_checkpoint_roundtrip(self):
        import torch
        torch.set_num_threads(1)
        for name, class_name in (("dffnet", "DFFNet"), ("m2heat", "M2Heat"), ("mft", "MFT")):
            for ms, sar, classes in ((6, 1, 5), (10, 4, 9)):
                for patch in (11, 13):
                    with self.subTest(model=name, channels=(ms, sar), patch=patch):
                        constructor = getattr(importlib.import_module(f"baselines.{name}.model"), class_name)
                        kwargs = dict(num_classes=classes, patch_size=patch, spectral_channels=ms, sar_channels=sar)
                        torch.manual_seed(3)
                        model = constructor(**kwargs)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        before = {id(p) for group in optimizer.param_groups for p in group["params"]}
                        x = torch.randn(2, 1, ms + sar, patch, patch)
                        output = model(x)
                        self.assertEqual(tuple(output.shape), (2, classes))
                        self.assertTrue(torch.isfinite(output).all())
                        torch.nn.functional.cross_entropy(output, torch.tensor([0, classes - 1])).backward()
                        trainable = [p for p in model.parameters() if p.requires_grad]
                        self.assertEqual(before, {id(p) for p in trainable})
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable))
                        buffer = io.BytesIO()
                        torch.save(model.state_dict(), buffer); buffer.seek(0)
                        restored = constructor(**kwargs)
                        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
                        model.eval(); restored.eval()
                        with torch.no_grad():
                            torch.testing.assert_close(model(x), restored(x))


if __name__ == "__main__":
    unittest.main()
