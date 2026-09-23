"""
Volume-wise PCA decomposition of k-space into principal-component bins.

Given a batch of slices that may come from several volumes (identified by `volume_id`), each volume's
slices are decomposed with a complex PCA over its own k-space: the slices are flattened to rows of an
n x (H*W) matrix, mean-centred, and thin-SVD'd. The principal components are grouped into bins and every
slice is reconstructed from each bin separately, giving one channel per bin. The channels of a slice sum
exactly to (slice - volume mean).

Binning rules
    equal_variance : cut the cumulative explained variance at 1/n_bins, 2/n_bins, ... (>= 1 PC per bin)
    equal_count    : the same number of PCs per bin

The basis (mean, singular vectors, bin boundaries) is computed under no_grad when `detach_basis=True`
(default): gradients then flow through the projection coefficients and reconstructions, which are plain
linear maps of the input, but never through the SVD itself (whose vector gradients are ~1/(s_i^2 - s_j^2)
and blow up in the flat noise tail of the spectrum).
"""
import torch

PCA_BIN_RULES = ("equal_variance", "equal_count")


def pca_bin_boundaries(var_ratio: torch.Tensor, n_bins: int, rule: str = "equal_variance"):
    """Return the list of (start, end) PC index ranges for `n_bins` bins over `var_ratio` (descending order)."""
    if rule not in PCA_BIN_RULES:
        raise ValueError(f"rule must be one of {PCA_BIN_RULES}, got '{rule}'")
    n = int(var_ratio.numel())
    if rule == "equal_count" or n < n_bins:
        cuts = [round(i * n / n_bins) for i in range(1, n_bins)]
    else:
        cum = torch.cumsum(var_ratio, 0)
        cuts = [int(torch.searchsorted(cum, torch.tensor(i / n_bins, dtype=cum.dtype, device=cum.device)).item()) + 1
                for i in range(1, n_bins)]
    # enforce >= 1 PC per bin when possible and monotone cuts
    bounds, prev = [], 0
    for i, c in enumerate(cuts):
        c = min(max(c, prev + 1), n - (n_bins - 1 - i))
        c = max(c, prev)
        bounds.append((prev, c))
        prev = c
    bounds.append((prev, n))
    return bounds


def volume_pca_channels(x: torch.Tensor, volume_id=None, n_bins: int = 3, rule: str = "equal_variance",
                        detach_basis: bool = True, center: bool = True, return_info: bool = False):
    """
    x          : [B, 1, H, W] complex k-space (any normalised domain)
    volume_id  : [B] int tensor / list; slices sharing an id form one volume. None -> whole batch is one volume.
    Returns
        channels : [B, n_bins, H, W]  bin reconstructions (sum over bins == x - mean)
        mean     : [B, 1, H, W]       per-volume mean k-space (zeros if center=False)
        info     : list of dicts per volume (bins, var per bin, n_components)   [only if return_info]
    """
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected [B, 1, H, W], got {tuple(x.shape)}")
    if not x.is_complex():
        raise ValueError("volume_pca_channels expects complex k-space")
    B, _, H, W = x.shape
    if volume_id is None:
        volume_id = torch.zeros(B, dtype=torch.long, device=x.device)
    volume_id = torch.as_tensor(volume_id, device=x.device).reshape(-1)
    if volume_id.numel() != B:
        raise ValueError(f"volume_id has {volume_id.numel()} entries for a batch of {B}")

    channels = x.new_zeros(B, n_bins, H, W)
    mean_out = x.new_zeros(B, 1, H, W)
    info = []
    for vid in torch.unique(volume_id):
        sel = torch.nonzero(volume_id == vid, as_tuple=False).reshape(-1)
        Xv = x[sel, 0].reshape(sel.numel(), -1)                                  # n x P (keeps grad)
        n = Xv.shape[0]
        if center:
            mean = Xv.mean(0, keepdim=True)
        else:
            mean = Xv.new_zeros(1, Xv.shape[1])
        Xc = Xv - mean
        ctx = torch.no_grad() if detach_basis else torch.enable_grad()
        with ctx:
            Xb = Xc.detach() if detach_basis else Xc
            if n < 2:
                Vh = None
            else:
                _, S, Vh = torch.linalg.svd(Xb, full_matrices=False)
                keep = S > S[0] * 1e-6 if S[0] > 0 else S > -1
                S, Vh = S[keep], Vh[keep]
                var_ratio = (S ** 2) / (S ** 2).sum().clamp_min(1e-30)
        if Vh is None or Vh.shape[0] < 1:
            # degenerate volume: everything in bin 0
            channels[sel, 0] = Xc.reshape(n, H, W)
            mean_out[sel, 0] = mean.reshape(1, H, W)
            info.append(dict(n_components=0, bins=[(0, 0)] * n_bins, var=[1.0] + [0.0] * (n_bins - 1)))
            continue
        bounds = pca_bin_boundaries(var_ratio, n_bins, rule)
        coeff = Xc @ Vh.conj().T                                                # n x K  (grad through Xc)
        for b, (s0, s1) in enumerate(bounds):
            if s1 > s0:
                channels[sel, b] = (coeff[:, s0:s1] @ Vh[s0:s1]).reshape(n, H, W)
        mean_out[sel, 0] = mean.reshape(1, H, W)
        info.append(dict(n_components=int(Vh.shape[0]), bins=bounds,
                         var=[float(var_ratio[s0:s1].sum()) for s0, s1 in bounds]))
    if return_info:
        return channels, mean_out, info
    return channels, mean_out
