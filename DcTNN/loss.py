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


class ComplexL2Loss(nn.Module):
    """L2 loss in the active complex domain: squared real diff + squared imag diff."""

    def forward(self, pred, gt, stats=None, mask=None):
        domain = _complex_target_domain(stats)
        _check_kspace_mask(mask, domain == "kspace", "complex_l2")
        gt_complex = _resolve_target(gt, domain)
        if not pred.is_complex() or not gt_complex.is_complex():
            raise ValueError("complex_l2 requires complex prediction and target tensors")
        return _reduce((pred.real - gt_complex.real) ** 2 + (pred.imag - gt_complex.imag) ** 2, mask)


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


def _radial_frequency_weight_map(x: torch.Tensor, m: float, gamma: float, r_cap=None) -> torch.Tensor:
    """
    Monotone radial k-space weight on the normalised square: w(r) = 1 + (m - 1) * (r / r_max)^gamma,
    with r from _normalized_radius_grid_like and r_max = sqrt(2) (the corner), so w = 1 at DC and
    w = m in the corners. m > 1 emphasises high frequencies, m == 1 is uniform.
    If r_cap is given, r_max = r_cap and the weight plateaus at m for r >= r_cap, so only the
    region inside r_cap is de-emphasised and everything outside it is weighted uniformly.
    """
    r_max = _FREQ_WEIGHT_R_MAX if r_cap is None else float(r_cap)
    ratio = (_normalized_radius_grid_like(x) / r_max).clamp(max=1.0)
    return 1.0 + (m - 1.0) * ratio.pow(gamma)


class FrequencyWeightedComplexL2Loss(nn.Module):
    """
    complex_l2 in k-space with each element weighted by a radial frequency weight.
    The weight is normalised to mean 1 over the reduced region (all k-space, or the
    unsampled region when a mask is given) so the loss scale matches complex_l2.
    """

    def __init__(self, weight_m: float = 5.0, weight_gamma: float = 1.0, weight_r_cap=None):
        super().__init__()
        self.weight_m = float(weight_m)
        self.weight_gamma = float(weight_gamma)
        self.weight_r_cap = None if weight_r_cap is None else float(weight_r_cap)

    def weight_map(self, x: torch.Tensor) -> torch.Tensor:
        return _radial_frequency_weight_map(x, self.weight_m, self.weight_gamma, self.weight_r_cap)

    def forward(self, pred, gt, stats=None, mask=None):
        if _complex_target_domain(stats) != "kspace":
            raise ValueError("freq_weighted_complex_l2 requires a k-space prediction (learning='k_space')")
        gt_complex = _resolve_target(gt, "kspace")
        if not pred.is_complex() or not gt_complex.is_complex():
            raise ValueError("freq_weighted_complex_l2 requires complex prediction and target tensors")
        elementwise = (pred.real - gt_complex.real) ** 2 + (pred.imag - gt_complex.imag) ** 2
        weight = self.weight_map(pred).to(elementwise.dtype)
        return _reduce(elementwise * weight, mask) / _reduce(weight.expand_as(elementwise), mask)


class ReconFormerMagnitudeL1Loss(nn.Module):
    def forward(self, pred, gt, stats=None, mask=None):
        _check_kspace_mask(mask, False, "reconformer_l1")
        gt_complex = _resolve_target(gt, "complex_image")
        if not pred.is_complex() or not gt_complex.is_complex():
            raise ValueError("reconformer_l1 requires complex prediction and target tensors")
        return F.l1_loss(pred.abs(), gt_complex.abs())


class PerpendicularLoss(nn.Module):
    """Perpendicular loss with a magnitude L1 term in the active complex domain."""

    def __init__(
        self,
        eps: float = 1e-8,
        magnitude_weighting: bool = False,
        magnitude_weight_m: float = 1.0,
        magnitude_weight_k: float = 0.103,
        magnitude_weight_p: float = 67.0,
    ):
        super().__init__()
        self.eps = eps
        self.magnitude_weighting = magnitude_weighting
        self.magnitude_weight_k = float(magnitude_weight_k)
        self.magnitude_weight_p = float(magnitude_weight_p)
        self.current_m = float(magnitude_weight_m)

    def set_current_m(self, value: float):
        self.current_m = float(value)

    def forward(self, pred, gt, stats=None, mask=None):
        domain = _complex_target_domain(stats)
        _check_kspace_mask(mask, domain == "kspace", "perpendicular_loss")
        gt_complex = _resolve_target(gt, domain)
        if not pred.is_complex() or not gt_complex.is_complex():
            raise ValueError("perpendicular_loss requires complex prediction and target tensors")

        cross = pred * gt_complex.conj()
        phi_hat = torch.angle(cross)
        perp = 0.5*torch.abs(pred * gt_complex.conj() - pred.conj() * gt_complex) / (pred.abs() + self.eps)
        target_abs = gt_complex.abs()
        branched = torch.where(phi_hat.abs() < (math.pi / 2), perp, 2 * target_abs - perp)
        magnitude_l1 = torch.abs(target_abs - pred.abs())
        if self.magnitude_weighting:
            magnitude_l1 = magnitude_l1 * _perpendicular_mag_weight_map(
                pred, m=self.current_m, k=self.magnitude_weight_k, p=self.magnitude_weight_p
            )
        return _reduce(branched + magnitude_l1, mask)


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
        "'complex_l1', 'complex_l2', 'freq_weighted_complex_l2', 'reconformer_l1', 'perpendicular_loss', "
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
