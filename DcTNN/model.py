import torch
from torch import nn
from .dc import FFT_DC, KSpace_DC, fft_2d
from .encoders import patchVIT, axVIT, imageEncoder, axialEncoder

# Re-export VIT classes so callers can import from DcTNN.model
__all__ = ['cascadeNet', 'patchVIT', 'axVIT', 'imageEncoder', 'axialEncoder']


class cascadeNet(nn.Module):
    """
    Defines a TNN that cascades denoising networks and applies data consistency.
    DC function is selected per stage: KSpace_DC for k-space stages, FFT_DC for image stages.
    Args:
        N (int)                     -       Image Size
        encList (array)             -       Should contain denoising network
        encArgs (array)             -       Contains dictionaries with args for encoders in encList
        lamb (bool)                 -       Whether or not to use a learned data consistency parameter
        k_space_learning            -       Bool or list[bool] — whether each stage operates in k-space
    """ 
    def __init__(self, N, encList, encArgs, lamb=True, k_space_learning=False):
        super(cascadeNet, self).__init__()
        if lamb:
            self.lamb = nn.Parameter(torch.ones(len(encList)) * 0.5)
        else:
            self.lamb = False
        self.scheduled_lamb = None
        self.N = N

        # k_space_learning: bool (all stages same) or list[bool] (per-stage)
        if isinstance(k_space_learning, (list, tuple)):
            assert len(k_space_learning) == len(encList), \
                "k_space_learning list length must match number of encoders"
            self._k_space_list = list(k_space_learning)
        else:
            self._k_space_list = [bool(k_space_learning)] * len(encList)

        # pick DC function per stage based on domain
        self._dc_funcs = [KSpace_DC if ks else FFT_DC for ks in self._k_space_list]

        transformers = []
        for i, enc in enumerate(encList):
            transformers.append(enc(N, **encArgs[i]))
        self.transformers = nn.ModuleList(transformers)

    def set_scheduled_lamb(self, value):
        self.scheduled_lamb = value

    def forward(self, xPrev, y, sampleMask):
        im = xPrev  # always image domain [B, C, H, W] entering each stage
        for i, transformer in enumerate(self.transformers):
            use_kspace = self._k_space_list[i]
            im_in = fft_2d(im) if use_kspace else im

            im_denoise = transformer(im_in)
            im_out = im_in + im_denoise

            if self.lamb is not False:
                lamb_i = self.lamb[i]
            elif self.scheduled_lamb is not None:
                lamb_i = self.scheduled_lamb
            else:
                lamb_i = None

            im = self._dc_funcs[i](im_out, y, sampleMask, lamb_i)
        return im
