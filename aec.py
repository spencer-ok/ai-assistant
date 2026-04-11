"""Acoustic Echo Cancellation using NLMS adaptive filter (vectorized)."""

import numpy as np
from numpy.fft import rfft, irfft


class EchoCanceller:
    """Frequency-domain adaptive filter for echo cancellation."""

    def __init__(self, block_size: int = 512, filter_blocks: int = 8,
                 mu: float = 0.4, eps: float = 1e-6):
        """
        Args:
            block_size: Process audio in chunks of this size.
            filter_blocks: Number of blocks in the filter (filter_blocks * block_size = echo tail).
                          At 16kHz with block_size=512, 8 blocks = 256ms.
            mu: Adaptation rate (0-1).
            eps: Regularization constant.
        """
        self.B = block_size
        self.K = filter_blocks
        self.mu = mu
        self.eps = eps
        self.fft_size = 2 * self.B

        # Filter weights in frequency domain
        self.W = np.zeros((self.K, self.fft_size // 2 + 1), dtype=np.complex128)
        # Reference signal buffer (K blocks)
        self.ref_blocks = np.zeros((self.K, self.fft_size // 2 + 1), dtype=np.complex128)
        # Overlap-save buffers
        self.ref_tail = np.zeros(self.B, dtype=np.float64)
        self.mic_tail = np.zeros(self.B, dtype=np.float64)

    def process_chunk(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Remove echo from a chunk of mic audio.

        Args:
            mic: Mic input (int16 or float). Will be converted to float64.
            ref: Reference/playback signal, same length as mic.

        Returns:
            Cleaned audio as float64 normalized to [-1, 1].
        """
        # Normalize to float64
        if mic.dtype == np.int16:
            mic = mic.astype(np.float64) / 32768.0
        if ref.dtype == np.int16:
            ref = ref.astype(np.float64) / 32768.0

        out = np.zeros(len(mic), dtype=np.float64)
        pos = 0

        while pos < len(mic):
            remaining = len(mic) - pos
            chunk_len = min(self.B, remaining)

            mic_chunk = np.zeros(self.B, dtype=np.float64)
            ref_chunk = np.zeros(self.B, dtype=np.float64)
            mic_chunk[:chunk_len] = mic[pos:pos + chunk_len]
            ref_chunk[:chunk_len] = ref[pos:pos + chunk_len]

            cleaned = self._process_block(mic_chunk, ref_chunk)
            out[pos:pos + chunk_len] = cleaned[:chunk_len]
            pos += chunk_len

        return out

    def _process_block(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        """Process one block through the adaptive filter."""
        # Build overlap-save input for reference
        ref_ext = np.concatenate([self.ref_tail, ref_block])
        self.ref_tail = ref_block.copy()

        # FFT of new reference block
        ref_fft = rfft(ref_ext, n=self.fft_size)

        # Shift reference blocks and insert new one
        self.ref_blocks = np.roll(self.ref_blocks, 1, axis=0)
        self.ref_blocks[0] = ref_fft

        # Compute echo estimate: sum of W[k] * X[k] in frequency domain
        Y = np.zeros(self.fft_size // 2 + 1, dtype=np.complex128)
        for k in range(self.K):
            Y += self.W[k] * self.ref_blocks[k]

        # Convert to time domain and take last B samples (overlap-save)
        y = irfft(Y, n=self.fft_size)
        echo_est = y[self.B:]

        # Error signal
        error = mic_block - echo_est

        # Build error FFT for weight update (zero-padded)
        mic_ext = np.concatenate([self.mic_tail, mic_block])
        self.mic_tail = mic_block.copy()
        error_ext = np.concatenate([np.zeros(self.B), error])
        error_fft = rfft(error_ext, n=self.fft_size)

        # Compute power for normalization
        power = self.eps
        for k in range(self.K):
            power += np.real(np.sum(self.ref_blocks[k] * np.conj(self.ref_blocks[k])))
        power /= self.K

        # Update weights
        for k in range(self.K):
            self.W[k] += (self.mu / power) * error_fft * np.conj(self.ref_blocks[k])

        return error
