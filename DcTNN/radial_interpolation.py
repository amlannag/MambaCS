import math

import torch


@torch.no_grad()
def radial_complex_interpolate(kspace, acquired_mask, strategy='linear', decay_length=8.0, distance_power=2.0):
    if strategy not in {'linear', 'exponential', 'inverse_distance', 'cartesian_linear'}:
        raise ValueError('Unknown interpolation strategy.')
    if not math.isfinite(decay_length) or decay_length <= 0:
        raise ValueError('decay_length must be positive and finite.')
    if not math.isfinite(distance_power) or distance_power <= 0:
        raise ValueError('distance_power must be positive and finite.')
    if kspace.ndim != 2 or not kspace.is_complex() or min(kspace.shape) < 2:
        raise ValueError('Expected a complex [H, W] k-space array with spatial sizes at least 2.')
    if acquired_mask.shape != kspace.shape or not torch.all((acquired_mask == 0) | (acquired_mask == 1)):
        raise ValueError('The acquired mask must be binary and match the k-space shape.')
    mask = acquired_mask.to(device=kspace.device, dtype=torch.bool)
    if not torch.equal(mask, mask[0:1].expand_as(mask)):
        raise ValueError('Interpolation requires a Cartesian full-column sampling mask.')
    height, width = kspace.shape
    cy, cx = height // 2, width // 2
    if not mask[cy, cx]:
        raise ValueError('The centered DC column must be acquired.')
    if not torch.isfinite(kspace[mask]).all():
        raise ValueError('Acquired k-space samples must be finite.')
    result = torch.where(mask, kspace, 0).clone()
    acquired_columns = torch.nonzero(mask[0], as_tuple=False).flatten()
    missing_columns = torch.nonzero(~mask[0], as_tuple=False).flatten()
    rows = torch.arange(height, device=kspace.device, dtype=kspace.real.dtype)
    row_indices = torch.arange(height, device=kspace.device)
    interpolated = torch.zeros_like(mask)
    endpoints = torch.zeros_like(mask)
    one_sided = torch.zeros_like(mask)
    for column in missing_columns.tolist():
        if strategy == 'cartesian_linear':
            left = acquired_columns[acquired_columns < column]
            right = acquired_columns[acquired_columns > column]
            bracketed = bool(left.numel() and right.numel())
            lo = int(left[-1]) if left.numel() else int(right[0])
            hi = int(right[0]) if right.numel() else lo
            weight = (column - lo) / (hi - lo) if hi != lo else 0.0
            result[:, column] = (1 - weight) * kspace[:, lo] + weight * kspace[:, hi]
            interpolated[:, column] = bracketed
            endpoints[:, column] = not bracketed
            one_sided[:, column] = not bracketed
            continue
        dx = column - cx
        candidates = acquired_columns[(acquired_columns - cx) * dx >= 0]
        t = (candidates.to(kspace.real.dtype) - cx) / dx
        crossings = cy + (rows[:, None] - cy) * t[None, :]
        valid = (crossings >= 0) & (crossings <= height - 1)
        has_upper = (valid & (t[None, :] >= 1)).any(dim=1)
        one_sided[:, column] = ~has_upper
        crossing_rows = crossings.clamp(0, height - 1)
        y0 = crossing_rows.floor().long()
        y1 = (y0 + 1).clamp_max(height - 1)
        row_weight = crossing_rows - y0
        samples = (1 - row_weight) * kspace[y0, candidates] + row_weight * kspace[y1, candidates]
        if strategy == 'linear':
            lower_options = torch.where(valid & (t[None, :] <= 1), t[None, :], -torch.inf)
            upper_options = torch.where(valid & (t[None, :] >= 1), t[None, :], torch.inf)
            lower = lower_options.argmax(dim=1)
            upper = upper_options.argmin(dim=1)
            upper = torch.where(has_upper, upper, lower)
            left_values = samples[row_indices, lower]
            right_values = samples[row_indices, upper]
            denominator = t[upper] - t[lower]
            radial_weight = torch.where(has_upper, (1 - t[lower]) / denominator.clamp_min(torch.finfo(kspace.real.dtype).eps), 0)
            result[:, column] = (1 - radial_weight) * left_values + radial_weight * right_values
            interpolated[:, column] = has_upper
            endpoints[:, column] = ~has_upper
        else:
            radius = ((rows - cy).square() + dx ** 2).sqrt()
            distances = radius[:, None] * (t[None, :] - 1).abs()
            logits = (-distances / decay_length if strategy == 'exponential'
                      else -distance_power * distances.clamp_min(torch.finfo(kspace.real.dtype).eps).log())
            weights = torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=1)
            result[:, column] = (weights * samples).sum(dim=1)
            interpolated[:, column] = True
    return result, {
        'strategy': strategy,
        'center': (cy, cx),
        'acquired_count': int(mask.sum()),
        'filled_count': int((~mask).sum()),
        'interpolated_count': int(interpolated.sum()),
        'endpoint_count': int(endpoints.sum()),
        'one_sided_count': int(one_sided.sum()),
        'interpolated_mask': interpolated,
        'endpoint_mask': endpoints,
        'one_sided_mask': one_sided,
    }
