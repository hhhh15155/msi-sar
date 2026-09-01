# VBE-Net Numerical Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the differentiable VBE-Net geometry head, render the architecture, and issue a numerical Go/No-Go decision before full-model training integration.

**Architecture:** `models/vbe_geometry.py` owns SPD matrix functions, grouped OAS Gaussians, Gaussian Bures distance, fixed-point barycenters, and the class-conditional variational energy. A synthetic benchmark checks identities, solver descent, autograd, latency, and memory at representative `[B=64,C=9,G=8,k=8]` shape without touching the dataset.

**Tech Stack:** Python 3, PyTorch, `unittest`, standard-library CLI/JSON/timing/XML, standalone SVG.

**Spec:** `docs/superpowers/specs/2026-09-01-vbe-net-design.md`

## Global Constraints

- Normal geometry executes in FP32; float64 is used for gradcheck and reference solves.
- Spectral inputs and outputs are symmetrized and eigenvalues are floored at `1e-4`.
- Default product geometry is `G=8`, `k=8`, normalized by `G*k=64`.
- Engineering solver defaults are `lambda_proto=1.0`, `tau_r=0.3`, `tau_c=0.1`, `inner_iters=3`, `outer_updates=1`.
- No attention, concatenation classifier, learned gate, auxiliary loss, or residual classifier is introduced.
- Tests use the repository's `unittest` convention and remain compatible with `python -m unittest discover -s tests -v`.
- This plan does not claim empirical accuracy. A Go result only authorizes the subsequent full-model/experiment-script plan.

## File Map

- Create `models/vbe_geometry.py`: numerical geometry and solver.
- Modify `models/__init__.py`: stable exports.
- Create `tests/test_vbe_geometry.py`: identities, descent, gradients, and stress cases.
- Create `scripts/prototype_vbe_solver.py`: deterministic benchmark and report writer.
- Create `tests/test_prototype_vbe_solver.py`: report contract tests.
- Create `docs/figures/vbe-net-architecture.svg`: static architecture diagram.
- Create `tests/test_vbe_architecture_figure.py`: SVG contract test.
- Create `docs/reports/vbe-prototype-report.json` and `.md`: measured evidence and decision.

---

### Task 1: SPD Primitives and Grouped Gaussian Distance

**Files:**
- Create: `models/vbe_geometry.py`
- Create: `tests/test_vbe_geometry.py`

**Interfaces:**
- Consumes: matrices `[...,k,k]`, tokens `[B,N,D]`, grouped means `[...,G,k]`, covariances `[...,G,k,k]`.
- Produces: `symmetrize`, `project_spd`, `matrix_sqrt_spd`, `matrix_invsqrt_spd`, `spd_from_raw_tril`, `estimate_grouped_gaussian`, `gaussian_bures_distance_sq`, `product_bures_distance_sq`.

- [ ] **Step 1: Write failing tests for SPD reconstruction and distance identities**

```python
class SPDAndDistanceTests(unittest.TestCase):
    def test_sqrt_and_invsqrt_reconstruct(self):
        torch.manual_seed(7)
        raw = torch.randn(4, 3, 3, dtype=torch.float64)
        spd = raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(3, dtype=torch.float64)
        root = matrix_sqrt_spd(spd, eps=1e-8)
        invroot = matrix_invsqrt_spd(spd, eps=1e-8)
        torch.testing.assert_close(root @ root, spd, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(invroot @ spd @ invroot, torch.eye(3).double().expand_as(spd), rtol=1e-6, atol=1e-7)

    def test_grouped_estimator_shapes_and_spd(self):
        estimate = estimate_grouped_gaussian(torch.randn(3, 121, 64), groups=8)
        self.assertEqual(estimate.mean.shape, (3, 8, 8))
        self.assertEqual(estimate.covariance.shape, (3, 8, 8, 8))
        self.assertTrue(torch.all((estimate.shrinkage >= 0) & (estimate.shrinkage <= 1)))
        self.assertGreaterEqual(torch.linalg.eigvalsh(estimate.covariance).min().item(), 1e-4 - 1e-6)

    def test_bures_identity_symmetry_and_scalar_formula(self):
        mean_a = torch.tensor([[[1.0]]], dtype=torch.float64)
        mean_b = torch.tensor([[[3.0]]], dtype=torch.float64)
        cov_a = torch.tensor([[[[4.0]]]], dtype=torch.float64)
        cov_b = torch.tensor([[[[9.0]]]], dtype=torch.float64)
        ab = gaussian_bures_distance_sq(mean_a, cov_a, mean_b, cov_b, eps=1e-8)
        ba = gaussian_bures_distance_sq(mean_b, cov_b, mean_a, cov_a, eps=1e-8)
        aa = gaussian_bures_distance_sq(mean_a, cov_a, mean_a, cov_a, eps=1e-8)
        torch.testing.assert_close(ab, torch.tensor([[5.0]], dtype=torch.float64))
        torch.testing.assert_close(ab, ba)
        self.assertLess(aa.abs().max().item(), 1e-8)
```

- [ ] **Step 2: Run the tests and confirm they fail because the module is absent**

Run: `python -m unittest tests.test_vbe_geometry.SPDAndDistanceTests -v`

Expected: import failure for `models.vbe_geometry`.

- [ ] **Step 3: Implement spectral maps, exact OAS formula, and Bures distance**

```python
def symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))

def _spectral_map(matrix: Tensor, transform, eps: float) -> Tensor:
    values, vectors = torch.linalg.eigh(symmetrize(matrix))
    mapped = transform(values.clamp_min(eps))
    return symmetrize((vectors * mapped.unsqueeze(-2)) @ vectors.transpose(-1, -2))

def project_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    return _spectral_map(matrix, lambda value: value, eps)

def matrix_sqrt_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    return _spectral_map(matrix, torch.sqrt, eps)

def matrix_invsqrt_spd(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    return _spectral_map(matrix, torch.rsqrt, eps)

def spd_from_raw_tril(raw: Tensor, eps: float = 1e-4, diagonal_floor: float = 1e-3) -> Tensor:
    lower = torch.tril(raw, diagonal=-1)
    diagonal = F.softplus(torch.diagonal(raw, dim1=-2, dim2=-1)) + diagonal_floor
    lower = lower + torch.diag_embed(diagonal)
    eye = torch.eye(raw.shape[-1], dtype=raw.dtype, device=raw.device)
    return symmetrize(lower @ lower.transpose(-1, -2) + eps * eye)

def estimate_grouped_gaussian(tokens: Tensor, groups: int = 8, eps: float = 1e-4) -> GroupedGaussian:
    batch, samples, channels = tokens.shape
    if channels % groups:
        raise ValueError("channels must be divisible by groups")
    dim = channels // groups
    grouped = tokens.reshape(batch, samples, groups, dim).transpose(1, 2)
    mean = grouped.mean(-2)
    centered = grouped - mean.unsqueeze(-2)
    empirical = centered.transpose(-1, -2) @ centered / samples
    trace = torch.diagonal(empirical, dim1=-2, dim2=-1).sum(-1)
    trace2 = empirical.square().sum((-2, -1))
    numerator = (1 - 2 / dim) * trace2 + trace.square()
    denominator = (samples + 1 - 2 / dim) * (trace2 - trace.square() / dim)
    rho = (numerator / denominator.clamp_min(1e-12)).clamp(0, 1)
    eye = torch.eye(dim, dtype=tokens.dtype, device=tokens.device)
    target = (trace / dim)[..., None, None] * eye
    covariance = (1-rho)[..., None, None] * empirical + rho[..., None, None] * target + eps * eye
    return GroupedGaussian(mean, symmetrize(covariance), rho)

def gaussian_bures_distance_sq(mean_a, covariance_a, mean_b, covariance_b, eps=1e-4):
    covariance_a = project_spd(covariance_a, eps)
    covariance_b = project_spd(covariance_b, eps)
    root_a = matrix_sqrt_spd(covariance_a, eps)
    cross_root = matrix_sqrt_spd(root_a @ covariance_b @ root_a, eps)
    mean_term = (mean_a - mean_b).square().sum(-1)
    covariance_term = torch.diagonal(covariance_a + covariance_b - 2 * cross_root, dim1=-2, dim2=-1).sum(-1)
    return (mean_term + covariance_term).clamp_min(0)

def product_bures_distance_sq(mean_a, covariance_a, mean_b, covariance_b, eps=1e-4, normalize=True):
    per_group = gaussian_bures_distance_sq(mean_a, covariance_a, mean_b, covariance_b, eps)
    total = per_group.sum(-1)
    return total / (mean_a.shape[-2] * mean_a.shape[-1]) if normalize else total
```

- [ ] **Step 4: Run focused and repository tests**

Run: `python -m unittest tests.test_vbe_geometry.SPDAndDistanceTests -v`

Expected: all Task 1 tests pass.

Run: `python -m unittest discover -s tests -v`

Expected: existing tests remain green.

- [ ] **Step 5: Commit Task 1**

```bash
git add models/vbe_geometry.py tests/test_vbe_geometry.py
git commit -m "Add VBE SPD and grouped Gaussian geometry"
```

---

### Task 2: Bures Barycenter, Variational Energy, and Autograd

**Files:**
- Modify: `models/vbe_geometry.py`
- Modify: `models/__init__.py`
- Modify: `tests/test_vbe_geometry.py`

**Interfaces:**
- Consumes: `means [...,M,G,k]`, `covariances [...,M,G,k,k]`, `weights [...,M]`, prototypes `[C,G,k]`, modalities `[B,M,G,k]`.
- Produces: `Barycenter`, `VBEResult`, `bures_barycenter`, `responsibility_from_distances`, `variational_bures_energy`.

- [ ] **Step 1: Write failing solver tests**

```python
class VariationalSolverTests(unittest.TestCase):
    def test_identical_barycenter_returns_input(self):
        mean = torch.tensor([[[1.0, -1.0]]], dtype=torch.float64)
        covariance = torch.tensor([[[[2.0, 0.3], [0.3, 1.0]]]], dtype=torch.float64)
        result = bures_barycenter(mean.unsqueeze(-3).expand(1, 3, 1, 2), covariance.unsqueeze(-4).expand(1, 3, 1, 2, 2), torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64), inner_iters=10, eps=1e-8)
        torch.testing.assert_close(result.mean, mean, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(result.covariance, covariance, rtol=1e-5, atol=1e-6)

    def test_responsibility_is_simplex_and_favors_near_modality(self):
        alpha = responsibility_from_distances(torch.tensor([[[0.1, 0.9], [1.2, 0.2]]]), tau_r=0.3)
        torch.testing.assert_close(alpha.sum(-1), torch.ones(1, 2))
        self.assertGreater(alpha[0, 0, 0], alpha[0, 0, 1])
        self.assertGreater(alpha[0, 1, 1], alpha[0, 1, 0])

    def test_solver_backward_is_finite(self):
        case = make_test_case(batch=2, classes=3, groups=2, dim=3, dtype=torch.float32, requires_grad=True)
        result = variational_bures_energy(*case, inner_iters=3, outer_updates=1)
        self.assertEqual(result.energy.shape, (2, 3))
        result.energy.sum().backward()
        for leaf in differentiable_leaves(case):
            self.assertIsNotNone(leaf.grad)
            self.assertTrue(torch.isfinite(leaf.grad).all())

    def test_small_float64_gradcheck(self):
        self.assertTrue(run_solver_gradcheck(seed=23, atol=3e-4, rtol=2e-3))
```

- [ ] **Step 2: Verify solver tests fail before implementation**

Run: `python -m unittest tests.test_vbe_geometry.VariationalSolverTests -v`

Expected: missing solver symbols.

- [ ] **Step 3: Implement fixed-point barycenter and closed-form responsibility**

```python
def bures_barycenter(means, covariances, weights, inner_iters=3, eps=1e-4):
    weights = weights / weights.sum(-1, keepdim=True)
    mean = (weights[..., :, None, None] * means).sum(-3)
    cov_weights = weights[..., :, None, None, None]
    covariance = project_spd((cov_weights * covariances).sum(-4), eps)
    for _ in range(inner_iters):
        root = matrix_sqrt_spd(covariance, eps)
        invroot = matrix_invsqrt_spd(covariance, eps)
        transported = matrix_sqrt_spd(root.unsqueeze(-3) @ covariances @ root.unsqueeze(-3), eps)
        average = (cov_weights * transported).sum(-4)
        covariance = project_spd(invroot @ average @ average @ invroot, eps)
    return Barycenter(mean, covariance)

def responsibility_from_distances(distances, tau_r=0.3, prior=None):
    if prior is None:
        prior = distances.new_full((distances.shape[-1],), 1 / distances.shape[-1])
    return torch.softmax(prior.log() - distances / tau_r, dim=-1)

class Barycenter(NamedTuple):
    mean: Tensor
    covariance: Tensor

class VBEResult(NamedTuple):
    energy: Tensor
    responsibility: Tensor
    fused_mean: Tensor
    fused_covariance: Tensor

def variational_bures_energy(
    prototype_mean,
    prototype_covariance,
    modality_mean,
    modality_covariance,
    lambda_proto=1.0,
    tau_r=0.3,
    inner_iters=3,
    outer_updates=1,
    eps=1e-4,
    prior=None,
):
    batch, modality_count, groups, group_dim = modality_mean.shape
    classes = prototype_mean.shape[0]
    pm = prototype_mean[None].expand(batch, classes, groups, group_dim)
    pc = prototype_covariance[None].expand(batch, classes, groups, group_dim, group_dim)
    mm = modality_mean[:, None].expand(batch, classes, modality_count, groups, group_dim)
    mc = modality_covariance[:, None].expand(batch, classes, modality_count, groups, group_dim, group_dim)
    if prior is None:
        prior = prototype_mean.new_full((modality_count,), 1 / modality_count)
    alpha = prior.expand(batch, classes, modality_count)

    def fuse(current_alpha):
        means = torch.cat((pm.unsqueeze(2), mm), dim=2)
        covariances = torch.cat((pc.unsqueeze(2), mc), dim=2)
        weights = torch.cat((torch.full_like(current_alpha[..., :1], lambda_proto), current_alpha), dim=-1)
        return bures_barycenter(means, covariances, weights, inner_iters, eps)

    fused = fuse(alpha)
    for _ in range(outer_updates):
        modality_distance = product_bures_distance_sq(fused.mean.unsqueeze(2), fused.covariance.unsqueeze(2), mm, mc, eps)
        alpha = responsibility_from_distances(modality_distance, tau_r, prior)
        fused = fuse(alpha)

    prototype_distance = product_bures_distance_sq(fused.mean, fused.covariance, pm, pc, eps)
    modality_distance = product_bures_distance_sq(fused.mean.unsqueeze(2), fused.covariance.unsqueeze(2), mm, mc, eps)
    safe_alpha = alpha.clamp_min(torch.finfo(alpha.dtype).tiny)
    safe_prior = prior.clamp_min(torch.finfo(prior.dtype).tiny)
    kl = (alpha * (safe_alpha.log() - safe_prior.log())).sum(-1)
    energy = lambda_proto * prototype_distance + (alpha * modality_distance).sum(-1) + tau_r * kl
    return VBEResult(energy, alpha, fused.mean, fused.covariance)
```

- [ ] **Step 4: Add reference coordinate-descent and repeated-spectrum tests**

The test helper evaluates `F(R0,alpha0)`, `F(R0,alpha1)`, and `F(R1,alpha1)` in float64 with 30 barycenter iterations. Assert each exact coordinate step raises energy by at most `1e-7`. Add a covariance spectrum `[1,1+1e-5,1+2e-5]` backward case and require finite gradients.

- [ ] **Step 5: Export stable interfaces and run all geometry tests**

```python
from .vbe_geometry import GroupedGaussian, VBEResult, estimate_grouped_gaussian, product_bures_distance_sq, variational_bures_energy
__all__ += ["GroupedGaussian", "VBEResult", "estimate_grouped_gaussian", "product_bures_distance_sq", "variational_bures_energy"]
```

Run: `python -m unittest tests.test_vbe_geometry -v`

Expected: identities, descent, backward, stress, and gradcheck all pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add models/vbe_geometry.py models/__init__.py tests/test_vbe_geometry.py
git commit -m "Add variational Bures energy solver"
```

---

### Task 3: Benchmark, Architecture SVG, and Artifact Tests

**Files:**
- Create: `scripts/prototype_vbe_solver.py`
- Create: `tests/test_prototype_vbe_solver.py`
- Create: `docs/figures/vbe-net-architecture.svg`
- Create: `tests/test_vbe_architecture_figure.py`

**Interfaces:**
- CLI: `--device --batch-size --classes --groups --group-dim --warmup --repeats --stress-batches --seed --json-out --markdown-out`.
- Report sections: `environment`, `shape`, `correctness`, `gradient`, `benchmarks`, `decision`.

- [ ] **Step 1: Write failing CLI/report and SVG tests**

```python
class PrototypeContractTests(unittest.TestCase):
    def test_tiny_report_schema(self):
        report = run_prototype(device="cpu", batch_size=2, classes=3, groups=2, group_dim=3, warmup=1, repeats=2, stress_batches=3, seed=31)
        self.assertEqual(set(report), {"environment", "shape", "correctness", "gradient", "benchmarks", "decision"})
        self.assertEqual(len(report["benchmarks"]), 6)
        self.assertIn(report["decision"]["status"], {"GO", "NO-GO"})

class FigureContractTests(unittest.TestCase):
    def test_svg_labels(self):
        root = ET.parse(ROOT / "docs/figures/vbe-net-architecture.svg").getroot()
        text = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("text"))
        for label in ("MS Patch", "SAR Patch", "Semi-shared Encoder", "Grouped OAS Gaussian", "Class Gaussian Prototypes", "Bures Barycenter", "Responsibility", "Variational Energy", "Logits", "Cross-Entropy"):
            self.assertIn(label, text)
```

- [ ] **Step 2: Run contract tests and confirm missing artifacts**

Run: `python -m unittest tests.test_prototype_vbe_solver tests.test_vbe_architecture_figure -v`

Expected: failures for missing script and SVG.

- [ ] **Step 3: Implement deterministic benchmark**

Benchmark these exact configurations:

```python
CONFIGURATIONS = [
    {"inner_iters": inner, "outer_updates": outer}
    for outer in (0, 1)
    for inner in (1, 3, 5)
]
```

Use `time.perf_counter_ns()` with CUDA synchronization where applicable. Measure forward and forward+backward latency after warm-up; report mean, median, standard deviation, min, max, and CUDA peak allocated bytes. Run `stress_batches` random forward/backward cases, count non-finite outputs/gradients, and compare `inner_iters=3` energy with a 30-iteration float64 reference on a tiny case. Include a diagonal Gaussian timing reference.

The decision is `GO` only when all identity flags pass, non-finite counts are zero, engineering/reference relative energy error is at most `5e-3`, and engineering coordinate-cycle energy increase is at most `1e-5`.

- [ ] **Step 4: Create standalone 1600x900 architecture SVG**

The SVG shows two separate modality lanes, shared pointwise weights as a dashed relation, grouped OAS Gaussians, class prototypes, and the exact solver sequence `alpha0 -> R0 -> alpha1 -> R1 -> E_c -> logits -> CE`. Use blue for MS, amber for SAR, violet for prototypes, and green for the solver. No module absent from the design spec may appear.

- [ ] **Step 5: Run contract tests and smoke benchmark**

Run: `python -m unittest tests.test_prototype_vbe_solver tests.test_vbe_architecture_figure -v`

Run: `python scripts/prototype_vbe_solver.py --device cpu --batch-size 2 --classes 3 --groups 2 --group-dim 3 --warmup 1 --repeats 2 --stress-batches 3 --json-out tmp/vbe-smoke.json --markdown-out tmp/vbe-smoke.md`

Expected: tests pass, command exits zero, and both reports are created.

- [ ] **Step 6: Commit Task 3**

```bash
git add scripts/prototype_vbe_solver.py tests/test_prototype_vbe_solver.py docs/figures/vbe-net-architecture.svg tests/test_vbe_architecture_figure.py
git commit -m "Add VBE prototype benchmark and architecture figure"
```

---

### Task 4: Representative Evidence and Training Decision

**Files:**
- Create: `docs/reports/vbe-prototype-report.json`
- Create: `docs/reports/vbe-prototype-report.md`
- Modify: `docs/superpowers/plans/2026-09-01-vbe-numerical-prototype.md`

**Interfaces:**
- Consumes: tested geometry module and benchmark.
- Produces: representative measurements and a gate that either starts or blocks full-model experiment integration.

- [ ] **Step 1: Run complete tests**

Run: `python -m unittest discover -s tests -v`

Expected: every existing and new test passes.

- [ ] **Step 2: Run representative benchmark**

Run: `python scripts/prototype_vbe_solver.py --device auto --batch-size 64 --classes 9 --groups 8 --group-dim 8 --warmup 10 --repeats 30 --stress-batches 100 --seed 20260901 --json-out docs/reports/vbe-prototype-report.json --markdown-out docs/reports/vbe-prototype-report.md`

- [ ] **Step 3: Audit exact decision fields**

Require `correctness.all_passed=true`, both gradient non-finite counts equal zero, `engineering_relative_energy_error<=0.005`, `engineering_energy_increase<=0.00001`, and an empty `decision.failed_checks` list for GO. Latency and memory are reported as machine evidence, not given an invented universal cutoff.

- [ ] **Step 4: Commit evidence and completed checkboxes**

```bash
git add docs/reports/vbe-prototype-report.json docs/reports/vbe-prototype-report.md docs/superpowers/plans/2026-09-01-vbe-numerical-prototype.md
git commit -m "Record VBE numerical prototype evidence"
```

- [ ] **Step 5: Follow the gate**

If GO, immediately create and execute a second plan for the full VBE-Net encoder and experiment scripts after inspecting the repository's actual dataset splits, channel counts, class counts, epochs, batch sizes, optimizer, scheduler, logging, and evaluation conventions. If NO-GO, correct the named numerical failure and rerun this task before writing training code.
