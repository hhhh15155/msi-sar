from __future__ import annotations

import unittest

import torch
from torch import nn

from models.vbe_net import VBENet, VBEModelOutput


def tiny_model(modality_dropout: float = 0.0) -> VBENet:
    return VBENet(
        ms_channels=10,
        sar_channels=4,
        num_classes=5,
        patch_size=11,
        width=16,
        depth=2,
        groups=4,
        expansion=2,
        inner_iters=1,
        outer_updates=1,
        modality_dropout=modality_dropout,
    )


def ms_patch(requires_grad: bool = False) -> torch.Tensor:
    return torch.randn(2, 10, 11, 11, requires_grad=requires_grad)


def sar_patch(requires_grad: bool = False) -> torch.Tensor:
    return torch.randn(2, 4, 11, 11, requires_grad=requires_grad)


class VBENetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)

    def test_encode_tokens_preserves_full_spatial_grid(self) -> None:
        model = tiny_model()

        ms_tokens, sar_tokens = model.encode_tokens(ms_patch(), sar_patch())

        self.assertEqual(ms_tokens.shape, (2, 121, 16))
        self.assertEqual(sar_tokens.shape, (2, 121, 16))

    def test_forward_returns_class_logits_and_details(self) -> None:
        model = tiny_model().eval()

        output = model(ms_patch(), sar_patch(), return_details=True)

        self.assertIsInstance(output, VBEModelOutput)
        self.assertEqual(output.logits.shape, (2, 5))
        self.assertEqual(output.energy.shape, (2, 5))
        self.assertEqual(output.responsibility.shape, (2, 5, 2))
        self.assertEqual(output.ms_shrinkage.shape, (2, 4))
        self.assertEqual(output.sar_shrinkage.shape, (2, 4))
        torch.testing.assert_close(output.logits, -output.energy / 0.1)
        torch.testing.assert_close(
            output.responsibility.sum(dim=-1),
            torch.ones(2, 5),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_plain_forward_returns_only_logits(self) -> None:
        logits = tiny_model().eval()(ms_patch(), sar_patch())

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, 5))

    def test_backward_reaches_inputs_and_prototypes(self) -> None:
        model = tiny_model().train()
        ms = ms_patch(requires_grad=True)
        sar = sar_patch(requires_grad=True)

        model(ms, sar).sum().backward()

        for gradient in (
            ms.grad,
            sar.grad,
            model.prototype_mean.grad,
            model.prototype_raw_tril.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_shared_pointwise_mixer_receives_each_lane_gradient(self) -> None:
        model = tiny_model().train()
        shared_weight = model.encoder.blocks[0].shared_pw1.weight

        ms_tokens, _ = model.encode_tokens(ms_patch(), sar_patch())
        ms_tokens.sum().backward()
        ms_gradient = shared_weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)

        _, sar_tokens = model.encode_tokens(ms_patch(), sar_patch())
        sar_tokens.sum().backward()
        sar_gradient = shared_weight.grad.detach().clone()

        self.assertGreater(ms_gradient.abs().sum().item(), 0.0)
        self.assertGreater(sar_gradient.abs().sum().item(), 0.0)

    def test_training_modality_dropout_removes_one_solver_observation(self) -> None:
        training_model = tiny_model(modality_dropout=1.0).train()
        evaluation_model = tiny_model(modality_dropout=1.0).eval()

        training_output = training_model(ms_patch(), sar_patch(), return_details=True)
        evaluation_output = evaluation_model(ms_patch(), sar_patch(), return_details=True)

        self.assertEqual(training_output.responsibility.shape[-1], 1)
        self.assertEqual(evaluation_output.responsibility.shape[-1], 2)

    def test_geometry_outputs_remain_float32_under_cpu_autocast(self) -> None:
        model = tiny_model().eval()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(ms_patch(), sar_patch(), return_details=True)

        self.assertEqual(output.energy.dtype, torch.float32)
        self.assertEqual(output.responsibility.dtype, torch.float32)

    def test_invalid_channel_or_spatial_shape_is_rejected(self) -> None:
        model = tiny_model()

        with self.assertRaisesRegex(ValueError, "MS channels"):
            model(torch.randn(2, 9, 11, 11), sar_patch())
        with self.assertRaisesRegex(ValueError, "SAR channels"):
            model(ms_patch(), torch.randn(2, 3, 11, 11))
        with self.assertRaisesRegex(ValueError, "spatial size"):
            model(torch.randn(2, 10, 9, 9), torch.randn(2, 4, 9, 9))

    def test_default_model_runs_production_shaped_cross_entropy_backward(self) -> None:
        model = VBENet(num_classes=8, modality_dropout=0.0).train()
        target = torch.tensor([1, 6])

        loss = nn.CrossEntropyLoss()(model(ms_patch(), sar_patch()), target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.prototype_mean.grad)
        self.assertTrue(torch.isfinite(model.prototype_mean.grad).all())


if __name__ == "__main__":
    unittest.main()
