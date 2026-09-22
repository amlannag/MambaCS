"""Loss functions for MambaCS reconstruction."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from normalizer import reconstruction_to_image_magnitude


_NORMALIZED_KSPACE_LOSS_NORMS = {"kspace_companding", "log_kspace"}
LOSS_FUNCTION_DOMAINS = ("all_kspace", "unsampled_kspace")


def _resolve_target(target, domain: str):
    if isinstance(target, dict):
        return target[domain]
    return target


def _use_normalized_kspace_loss(stats) -> bool:
    return bool(stats and stats.get("normalization") in _NORMALIZED_KSPACE_LOSS_NORMS)


def _complex_target_domain(stats) -> str:
    return "complex_image" if stats and stats.get("prediction_domain") == "complex_image" else "kspace"


def _check_kspace_mask(mask, in_kspace: bool, name: str):
    if mask is not None and not in_kspace:
        raise ValueError(
            f"loss_function_domain='unsampled_kspace' requires {name} to operate in k-space, "
            "but the prediction is in the image domain"
        )


def _reduce(elementwise: torch.Tensor, mask=None) -> torch.Tensor:
    """
    Mean over all elements (mask=None) or over the unsampled k-space locations only
    (mask broadcastable to `elementwise`, 1 = sampled, 0 = unsampled).
    """
    if mask is None:
        return elementwise.mean()
    weight = (1.0 - mask).to(elementwise.dtype).expand_as(elementwise)
    return (elementwise * weight).sum() / weight.sum().clamp_min(1.0)


def _to_magnitude(x, stats=None):
    """
    Bring x into the real magnitude image domain.
    - complex k-space: inverse FFT then magnitude
    - complex image: magnitude directly
    - real image: pass through as-is
    """
    if x.is_complex():
        if _use_normalized_kspace_loss(stats):
            return x.abs()
        return reconstruction_to_image_magnitude(x, stats)
    return x


def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)


def _radial_distance_grid_like(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected a [B, C, H, W] tensor, got shape {tuple(x.shape)}")
    _, _, h, w = x.shape
    ys = torch.arange(h, device=x.device, dtype=x.real.dtype if x.is_complex() else x.dtype)
    xs = torch.arange(w, device=x.device, dtype=ys.dtype)
    cy = h // 2
    cx = w // 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return dist.unsqueeze(0).unsqueeze(0)


def _perpendicular_mag_weight_map(x: torch.Tensor, m: float, k: float, p: float) -> torch.Tensor:
    dist = _radial_distance_grid_like(x)
    left = (m - 1.0) * torch.exp(k * (dist - p)) + 1.0
    ones = torch.ones_like(left)
    return torch.where(dist < p, left, ones)


class MagnitudeImageLoss(nn.Module):
    """MSE in the normalised magnitude image domain."""
    def forward(self, pred, gt, stats=None, mask=None):
        in_kspace = _use_normalized_kspace_loss(stats)
        _check_kspace_mask(mask, in_kspace, "l2")
        gt_tensor = _resolve_target(gt, "kspace" if in_kspace else "image")
        return _reduce((_to_magnitude(pred, stats) - _to_magnitude(gt_tensor, stats)) ** 2, mask)


class MagnitudeL1Loss(nn.Module):
    """L1 loss in the normalised magnitude image domain."""
    def forward(self, pred, gt, stats=None, mask=None):
        in_kspace = _use_normalized_kspace_loss(stats)
        _check_kspace_mask(mask, in_kspace, "l1")
        gt_tensor = _resolve_target(gt, "kspace" if in_kspace else "image")
        return _reduce(torch.abs(_to_magnitude(pred, stats) - _to_magnitude(gt_tensor, stats)), mask)


class ComplexL1Loss(nn.Module):
    """L1 loss in the active complex domain: |Re diff| + |Im diff|."""

    def forward(self, pred, gt, stats=None, mask=None):
        domain = _complex_target_domain(stats)
        _check_kspace_mask(mask, domain == "kspace", "complex_l1")
        gt_complex = _resolve_target(gt, domain)
        if not pred.is_complex() or not gt_complex.is_complex():
            raise ValueError("complex_l1 requires complex prediction and target tensors")
        return _reduce(torch.abs(pred.real - gt_complex.real) + torch.abs(pred.imag - gt_complex.imag), mask)


def _complex_pair(pred, gt, stats, mask, name):
    """Resolve the complex target for a complex loss and run the shared validity checks."""
    domain = _complex_target_domain(stats)
    _check_kspace_mask(mask, domain == "kspace", name)
    gt_complex = _resolve_target(gt, domain)
    if not pred.is_complex() or not gt_complex.is_complex():
        raise ValueError(f"{name} requires complex prediction and target tensors")
    return gt_complex


def _squared_error(pred, gt_complex):
    return (pred.real - gt_complex.real) ** 2 + (pred.imag - gt_complex.imag) ** 2


class _ElementwiseComplexLoss(nn.Module):
    """
    Base for complex losses that are the region mean of a per-element term:
    `elementwise` returns the per-cell term so that forward == _reduce(elementwise, mask).
    """
    name = "complex"

    def elementwise(self, pred, gt, stats=None, mask=None):
        raise NotImplementedError

    def forward(self, pred, gt, stats=None, mask=None):
        return _reduce(self.elementwise(pred, gt, stats, mask), mask)


class ComplexL2Loss(_ElementwiseComplexLoss):
    """L2 loss in the active complex domain: squared real diff + squared imag diff."""
    name = "complex_l2"

    def elementwise(self, pred, gt, stats=None, mask=None):
        return _squared_error(pred, _complex_pair(pred, gt, stats, mask, self.name))


class ComplexL2NMSELoss(_ElementwiseComplexLoss):
    """
    Normalised complex L2 in the active complex domain, per sample:
        sum |pred - gt|^2 / sum |gt|^2
    over all elements, or over the unsampled k-space locations only when a mask is given.
    Scale-free, so slices with very different k-space energy contribute comparably.
    The per-element term is |pred - gt|^2 divided by the sample's mean |gt|^2 over the region.
    """
    name = "complex_l2_nmse"

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def elementwise(self, pred, gt, stats=None, mask=None):
        gt_complex = _complex_pair(pred, gt, stats, mask, self.name)
        error = _squared_error(pred, gt_complex)
        energy = gt_complex.real ** 2 + gt_complex.imag ** 2
        weight = torch.ones_like(error) if mask is None else (1.0 - mask).to(error.dtype).expand_as(error)
        dims = tuple(range(1, error.ndim))
        mean_energy = (energy * weight).sum(dims, keepdim=True) / weight.sum(dims, keepdim=True).clamp_min(1.0)
        return error / mean_energy.clamp_min(self.eps)


class PointwiseNormalizedComplexL2Loss(_ElementwiseComplexLoss):
    """
    Pixel-by-pixel normalised complex L2:  mean |pred - gt|^2 / (|gt| + eps).
    Each k-space cell's squared error is divided by the magnitude of its own target, so low-energy
    (high-frequency) cells are not swamped by the bright centre.
    """
    name = "complex_l2_pointwise_normalized"

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def elementwise(self, pred, gt, stats=None, mask=None):
        gt_complex = _complex_pair(pred, gt, stats, mask, self.name)
        return _squared_error(pred, gt_complex) / (gt_complex.abs() + self.eps)


class ComplexBerhuLoss(_ElementwiseComplexLoss):
    """
    Plain reverse-Huber (Berhu) loss on the complex error modulus e = |pred - gt|:
        |e|                      if |e| <= delta      (linear for small errors)
        (e^2 + delta^2) / (2 delta)   otherwise        (quadratic for large errors)
    The two branches meet with matching value and slope at |e| = delta. With delta = 1 this is the
    smooth version of max(|e|, e^2). Unlike Huber it is L1-like near zero and L2-like in the tails,
    so large errors are penalised quadratically while small ones keep a constant-magnitude gradient.
    """
    name = "complex_berhu"

    def __init__(self, delta: float = 1.0):
        super().__init__()
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = float(delta)

    def elementwise(self, pred, gt, stats=None, mask=None):
        gt_complex = _complex_pair(pred, gt, stats, mask, self.name)
        e = (pred - gt_complex).abs()
        quadratic = (e ** 2 + self.delta ** 2) / (2.0 * self.delta)
        return torch.where(e <= self.delta, e, quadratic)


def _normalized_radius_grid_like(x: torch.Tensor) -> torch.Tensor:
    """
    Radius of every k-space cell on the normalised square [-1, 1]^2 centred at (0, 0):
    x = (col - W//2) / (W/2), y = (row - H//2) / (H/2), r = sqrt(x^2 + y^2).
    r = 0 at DC, r = 1 at the edge midpoints, r = sqrt(2) at the corners.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected a [B, C, H, W] tensor, got shape {tuple(x.shape)}")
    _, _, h, w = x.shape
    dtype = x.real.dtype if x.is_complex() else x.dtype
    ys = (torch.arange(h, device=x.device, dtype=dtype) - h // 2) / (h / 2)
    xs = (torch.arange(w, device=x.device, dtype=dtype) - w // 2) / (w / 2)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(0).unsqueeze(0)


_FREQ_WEIGHT_R_MAX = math.sqrt(2.0)   # corner of the normalised [-1, 1]^2 square


FREQ_WEIGHT_FORMS = ("power", "exp", "exp_saturating", "gauss_dip", "rings")


def _normalized_kx_grid_like(x: torch.Tensor) -> torch.Tensor:
    """|kx| of every cell on the normalised square: 0 at the centre column, 1 at the left/right edge."""
    if x.ndim != 4:
        raise ValueError(f"Expected a [B, C, H, W] tensor, got shape {tuple(x.shape)}")
    _, _, h, w = x.shape
    dtype = x.real.dtype if x.is_complex() else x.dtype
    xs = ((torch.arange(w, device=x.device, dtype=dtype) - w // 2) / (w / 2)).abs()
    return xs.expand(h, w).unsqueeze(0).unsqueeze(0)


def _piecewise_rings(coord: torch.Tensor, edges, weights) -> torch.Tensor:
    if edges is None or weights is None or len(weights) != len(edges) - 1:
        raise ValueError("rings need edges (n+1 ascending values) and weights (n values)")
    w = torch.ones_like(coord)
    for lo, hi, wt in zip(edges[:-1], edges[1:], weights):
        w = torch.where((coord >= float(lo)) & (coord < float(hi)), torch.full_like(coord, float(wt)), w)
    return w


def _radial_frequency_weight_map(x: torch.Tensor, m: float, gamma: float, r_cap=None,
                                 form: str = "power", a: float = 1.0, r_stop=None,
                                 ring_edges=None, ring_weights=None,
                                 kx_ring_edges=None, kx_ring_weights=None) -> torch.Tensor:
    """
    Radial k-space weight w(r) on the normalised square (r from _normalized_radius_grid_like:
    0 at DC, 1 at the edge midpoints, sqrt(2) in the corners). The loss divides by mean(w), so only
    the SHAPE matters; the continuous forms have w(0) = 1 before normalisation.

    form="power"          w = 1 + (m - 1) * (r / r_max)^gamma,  r_max = sqrt2 or r_cap (plateau at m beyond r_cap)
    form="exp"            w = exp(a * r);  a > 0 boosts high frequencies, a < 0 suppresses them
    form="exp_saturating" w = 1 + (m - 1) * (1 - exp(-a * r))   smooth rise from 1 toward m (63% at r = 1/a)
    form="gauss_dip"      w = 1 - (1 - 1/m) * exp(-(a * r)^2)   1/m at DC rising to 1; only the centre is de-emphasised
    form="rings"          piecewise constant: w = ring_weights[i] for ring_edges[i] <= r < ring_edges[i+1]
                          (len(ring_weights) == len(ring_edges) - 1; r beyond the last edge -> 1)

    kx_ring_edges / kx_ring_weights (any form, optional): an additional piecewise-constant factor on the
    normalised column distance |kx| (0 = centre column, 1 = edge), multiplied into w. Useful because the
    Cartesian mask varies along kx only, so the unmeasured region is a function of |kx| rather than r.

    r_stop (any form): for r >= r_stop the weight is reset to exactly 1, i.e. the weighting only acts
    inside r_stop. This is a hard step unless the form is already ~1 there.
    """
    if form not in FREQ_WEIGHT_FORMS:
        raise ValueError(f"form must be one of {FREQ_WEIGHT_FORMS}, got '{form}'")
    r = _normalized_radius_grid_like(x)
    if form == "power":
        r_max = _FREQ_WEIGHT_R_MAX if r_cap is None else float(r_cap)
        w = 1.0 + (m - 1.0) * (r / r_max).clamp(max=1.0).pow(gamma)
    elif form == "exp":
        w = torch.exp((a * r).clamp(max=80.0))
    elif form == "exp_saturating":
        w = 1.0 + (m - 1.0) * (1.0 - torch.exp(-a * r))
    elif form == "gauss_dip":
        w = 1.0 - (1.0 - 1.0 / m) * torch.exp(-(a * r) ** 2)
    else:
        w = _piecewise_rings(r, ring_edges, ring_weights)
    if kx_ring_edges is not None or kx_ring_weights is not None:
        w = w * _piecewise_rings(_normalized_kx_grid_like(x), kx_ring_edges, kx_ring_weights)
    if r_stop is not None:
        w = torch.where(r < float(r_stop), w, torch.ones_like(w))
    return w


class FrequencyWeightedComplexL2Loss(_ElementwiseComplexLoss):
    """
    complex_l2 in k-space with each element weighted by a radial frequency weight.
    The weight is normalised to mean 1 over the reduced region (all k-space, or the
    unsampled region when a mask is given) so the loss scale matches complex_l2.
    """
    name = "freq_weighted_complex_l2"

    def __init__(self, weight_m: float = 5.0, weight_gamma: float = 1.0, weight_r_cap=None,
                 weight_form: str = "power", weight_a: float = 1.0, weight_r_stop=None,
                 weight_ring_edges=None, weight_ring_weights=None,
                 weight_kx_ring_edges=None, weight_kx_ring_weights=None):
        super().__init__()
        if weight_form not in FREQ_WEIGHT_FORMS:
            raise ValueError(f"weight_form must be one of {FREQ_WEIGHT_FORMS}, got '{weight_form}'")
        self.weight_kx_ring_edges = None if weight_kx_ring_edges is None else [float(v) for v in weight_kx_ring_edges]
        self.weight_kx_ring_weights = None if weight_kx_ring_weights is None else [float(v) for v in weight_kx_ring_weights]
        self.weight_m = float(weight_m)
        self.weight_gamma = float(weight_gamma)
        self.weight_r_cap = None if weight_r_cap is None else float(weight_r_cap)
        self.weight_form = weight_form
        self.weight_a = float(weight_a)
        self.weight_r_stop = None if weight_r_stop is None else float(weight_r_stop)
        self.weight_ring_edges = None if weight_ring_edges is None else [float(v) for v in weight_ring_edges]
        self.weight_ring_weights = None if weight_ring_weights is None else [float(v) for v in weight_ring_weights]

    def weight_map(self, x: torch.Tensor) -> torch.Tensor:
        return _radial_frequency_weight_map(x, self.weight_m, self.weight_gamma, self.weight_r_cap,
                                            form=self.weight_form, a=self.weight_a, r_stop=self.weight_r_stop,
                                            ring_edges=self.weight_ring_edges, ring_weights=self.weight_ring_weights,
                                            kx_ring_edges=self.weight_kx_ring_edges, kx_ring_weights=self.weight_kx_ring_weights)

    def elementwise(self, pred, gt, stats=None, mask=None):
        if _complex_target_domain(stats) != "kspace":
            raise ValueError("freq_weighted_complex_l2 requires a k-space prediction (learning='k_space')")
        error = _squared_error(pred, _complex_pair(pred, gt, stats, mask, self.name))
        weight = self.weight_map(pred).to(error.dtype).expand_as(error)
        return error * weight / _reduce(weight, mask)


class ReconFormerMagnitudeL1Loss(nn.Module):
    def forward(self, pred, gt, stats=None, mask=None):
        _check_kspace_mask(mask, False, "reconformer_l1")
        gt_complex = _resolve_target(gt, "complex_image")
        if not pred.is_complex() or not gt_complex.is_complex():
            raise ValueError("reconformer_l1 requires complex prediction and target tensors")
        return F.l1_loss(pred.abs(), gt_complex.abs())


class PerpendicularLoss(_ElementwiseComplexLoss):
    """
    Perpendicular loss (branched phase term) plus a magnitude term in the active complex domain.

    phase term (Terpstra et al. eq. 3-4):  perp = |gt| |sin dphi| = |Im(pred conj(gt))| / |pred|,
        branched = perp if |dphi| < 90 deg else 2|gt| - perp.
    phase_scale:    "none" -> branched as published (scale-free in |pred|; gradient ~ |gt|/|pred|)
                    "pred" -> |pred| * branched  (= |Im(pred conj(gt))| for |dphi| < 90 deg): removes the
                              1/|pred| normalisation so a wrong phase costs in proportion to the asserted
                              magnitude, the gradient scales with |gt| (SNR-weighted) and the term -> 0 at pred = 0.
    r_boundary / branch_multiplier: optional radial gating of the phase term on the normalised k-space radius
        (0 = DC, 1 = edge midpoint, sqrt2 = corner): cells with r >= r_boundary have their phase term multiplied
        by branch_multiplier (0 = phase loss off outside the boundary); cells inside keep multiplier 1.
        r_boundary=None disables the gating.
    phase_norm:     "l1" -> branched term as is;  "l2" -> branched term squared (applied after phase_scale,
                    before the radial multiplier). With phase_scale="none", "l2" gives |gt|^2 sin^2(dphi), which is
                    quadratic in the cell's scale like the L2 magnitude term.
    magnitude_norm: "l1" -> | |gt| - |pred| |   (original),  "l2" -> (|gt| - |pred|)^2.
    """
    name = "perpendicular_loss"

    def __init__(
        self,
        eps: float = 1e-8,
        magnitude_weighting: bool = False,
        magnitude_weight_m: float = 1.0,
        magnitude_weight_k: float = 0.103,
        magnitude_weight_p: float = 67.0,
        magnitude_norm: str = "l1",
        phase_scale: str = "none",
        phase_norm: str = "l1",
        r_boundary=None,
        branch_multiplier: float = 1.0,
    ):
        super().__init__()
        if magnitude_norm not in {"l1", "l2"}:
            raise ValueError(f"magnitude_norm must be 'l1' or 'l2', got '{magnitude_norm}'")
        if phase_scale not in {"none", "pred"}:
            raise ValueError(f"phase_scale must be 'none' or 'pred', got '{phase_scale}'")
        if phase_norm not in {"l1", "l2"}:
            raise ValueError(f"phase_norm must be 'l1' or 'l2', got '{phase_norm}'")
        self.eps = eps
        self.magnitude_norm = magnitude_norm
        self.phase_scale = phase_scale
        self.phase_norm = phase_norm
        self.r_boundary = None if r_boundary is None else float(r_boundary)
        self.branch_multiplier = float(branch_multiplier)
        self.magnitude_weighting = magnitude_weighting
        self.magnitude_weight_k = float(magnitude_weight_k)
        self.magnitude_weight_p = float(magnitude_weight_p)
        self.current_m = float(magnitude_weight_m)

    def set_current_m(self, value: float):
        self.current_m = float(value)

    def phase_multiplier_map(self, x: torch.Tensor) -> torch.Tensor:
        """[1, 1, H, W] radial multiplier applied to the phase term (1 inside r_boundary, branch_multiplier outside)."""
        r = _normalized_radius_grid_like(x)
        if self.r_boundary is None:
            return torch.ones_like(r)
        return torch.where(r < self.r_boundary, torch.ones_like(r), torch.full_like(r, self.branch_multiplier))

    def components(self, pred, gt, stats=None, mask=None):
        """Per-cell (phase term, magnitude term); their sum is `elementwise`."""
        gt_complex = _complex_pair(pred, gt, stats, mask, self.name)

        cross = pred * gt_complex.conj()
        phi_hat = torch.angle(cross)
        perp = 0.5*torch.abs(pred * gt_complex.conj() - pred.conj() * gt_complex) / (pred.abs() + self.eps)
        target_abs = gt_complex.abs()
        branched = torch.where(phi_hat.abs() < (math.pi / 2), perp, 2 * target_abs - perp)
        if self.phase_scale == "pred":
            branched = branched * pred.abs()
        if self.phase_norm == "l2":
            branched = branched ** 2
        if self.r_boundary is not None:
            branched = branched * self.phase_multiplier_map(pred).to(branched.dtype)
        magnitude_diff = target_abs - pred.abs()
        magnitude_term = magnitude_diff ** 2 if self.magnitude_norm == "l2" else torch.abs(magnitude_diff)
        if self.magnitude_weighting:
            magnitude_term = magnitude_term * _perpendicular_mag_weight_map(
                pred, m=self.current_m, k=self.magnitude_weight_k, p=self.magnitude_weight_p
            )
        return branched, magnitude_term

    def elementwise(self, pred, gt, stats=None, mask=None):
        phase_term, magnitude_term = self.components(pred, gt, stats, mask)
        return phase_term + magnitude_term


def _loraks_offsets(radius: int):
    return [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius]


class LoraksCLoss(nn.Module):
    """
    LORAKS C-matrix (support-constraint) low-rank penalty on the predicted k-space.

    Row n of P_C(k) is the radius-R k-space neighbourhood around point n (Haldar 2014, Eq. 5).
    The penalty is the energy of the singular values beyond `rank`,
        ||P_C(k) - SVD_rank(P_C(k))||_F^2 = sum_{i > rank} sigma_i^2,
    computed from the small Nr x Nr Gram matrix C^H C so the full SVD is never needed.
    With normalize="ratio" this is divided by ||P_C(k)||_F^2 (scale-free, in [0, 1]);
    with normalize="mean" it is divided by the number of matrix entries.

    This is an unsupervised prior: it only looks at `pred`, never at `gt`.
    """

    def __init__(self, radius: int = 3, rank=None, normalize: str = "ratio", eps: float = 1e-8):
        super().__init__()
        self.radius = int(radius)
        self.offsets = _loraks_offsets(self.radius)
        self.Nr = len(self.offsets)
        self.rank = int(rank) if rank is not None else max(1, self.Nr // 2)
        if not 0 < self.rank < self.Nr:
            raise ValueError(f"loraks rank must be in (0, {self.Nr}) for radius {self.radius}, got {self.rank}")
        if normalize not in {"ratio", "mean"}:
            raise ValueError("normalize must be 'ratio' or 'mean'")
        self.normalize = normalize
        self.eps = eps

    def c_matrix(self, kspace: torch.Tensor) -> torch.Tensor:
        """[B, H, W] complex k-space -> [B, K, Nr] C matrix (K = valid centres)."""
        R = self.radius
        H, W = kspace.shape[-2:]
        cols = [kspace[..., R + dx:H - R + dx, R + dy:W - R + dy].reshape(kspace.shape[0], -1)
                for dx, dy in self.offsets]
        return torch.stack(cols, dim=-1)

    def forward(self, pred, gt=None, stats=None, mask=None):
        if not pred.is_complex():
            raise ValueError("loraks_c requires a complex prediction tensor")
        if _complex_target_domain(stats) == "complex_image":
            from DcTNN.dc import fft_2d
            pred = fft_2d(pred)
        kspace = pred.reshape(-1, *pred.shape[-2:])
        if kspace.dtype != torch.complex128:
            kspace = kspace.to(torch.complex64)
        C = self.c_matrix(kspace)                                 # [B, K, Nr]
        gram = C.transpose(-1, -2).conj() @ C                     # [B, Nr, Nr]
        gram = gram + self.eps * torch.eye(self.Nr, device=gram.device, dtype=gram.dtype)
        sigma_sq = torch.linalg.eigvalsh(gram)                    # ascending, real
        tail = sigma_sq[:, :self.Nr - self.rank].sum(dim=-1)
        if self.normalize == "ratio":
            per_sample = tail / (sigma_sq.sum(dim=-1) + self.eps)
        else:
            per_sample = tail / C.shape[-1] / C.shape[-2]
        return per_sample.mean()


class ComplexL2LoraksLoss(nn.Module):
    """complex_l2 data term plus weight * LORAKS C low-rank penalty on the prediction."""

    def __init__(self, weight: float = 0.05, radius: int = 3, rank=None, normalize: str = "ratio"):
        super().__init__()
        self.weight = float(weight)
        self.l2 = ComplexL2Loss()
        self.loraks = LoraksCLoss(radius=radius, rank=rank, normalize=normalize)

    def forward(self, pred, gt, stats=None, mask=None):
        return self.l2(pred, gt, stats=stats, mask=mask) + self.weight * self.loraks(pred, stats=stats)


def build_loss(loss_type: str, **kwargs) -> nn.Module:
    """Factory for magnitude-domain reconstruction losses."""
    loss_type = loss_type.lower()
    if loss_type in {"l1", "image_domain_l1"}:
        return MagnitudeL1Loss()
    if loss_type in {"l2", "image_domain_l2"}:
        return MagnitudeImageLoss()
    if loss_type == "complex_l1":
        return ComplexL1Loss()
    if loss_type == "complex_l2":
        return ComplexL2Loss()
    if loss_type == "complex_l2_nmse":
        return ComplexL2NMSELoss()
    if loss_type == "complex_l2_pointwise_normalized":
        return PointwiseNormalizedComplexL2Loss(**kwargs)
    if loss_type == "complex_berhu":
        return ComplexBerhuLoss(**kwargs)
    if loss_type == "freq_weighted_complex_l2":
        return FrequencyWeightedComplexL2Loss(**kwargs)
    if loss_type == "reconformer_l1":
        return ReconFormerMagnitudeL1Loss()
    if loss_type == "perpendicular_loss":
        return PerpendicularLoss(**kwargs)
    if loss_type == "loraks_c":
        return LoraksCLoss(**kwargs)
    if loss_type == "complex_l2_loraks":
        return ComplexL2LoraksLoss(**kwargs)
    raise ValueError(
        "Unknown loss_type "
        f"'{loss_type}'. Choose from: ['l1', 'l2', 'image_domain_l1', 'image_domain_l2', "
        "'complex_l1', 'complex_l2', 'complex_l2_nmse', 'complex_l2_pointwise_normalized', 'complex_berhu', 'freq_weighted_complex_l2', "
        "'reconformer_l1', 'perpendicular_loss', "
        "'loraks_c', 'complex_l2_loraks']"
    )


class SSIMLoss(nn.Module):
    """
    1 - SSIM, computed in the magnitude image domain.
    Uses an 11×11 Gaussian window (sigma=1.5), standard parameters.
    """

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        kernel = _gaussian_kernel(self.kernel_size, self.sigma, device=x.device)
        pad = self.kernel_size // 2

        mu_x  = F.conv2d(x, kernel, padding=pad)
        mu_y  = F.conv2d(y, kernel, padding=pad)
        mu_xx = F.conv2d(x * x, kernel, padding=pad)
        mu_yy = F.conv2d(y * y, kernel, padding=pad)
        mu_xy = F.conv2d(x * y, kernel, padding=pad)

        sigma_x  = mu_xx - mu_x ** 2
        sigma_y  = mu_yy - mu_y ** 2
        sigma_xy = mu_xy - mu_x * mu_y

        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return (num / den).mean()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, stats=None, mask=None) -> torch.Tensor:
        _check_kspace_mask(mask, False, "ssim")
        gt_image = _resolve_target(gt, "image")
        p = _to_magnitude(pred, stats)
        g = _to_magnitude(gt_image)
        return 1.0 - self._ssim(p, g)
