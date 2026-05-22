from PIL import Image, ImageFile
import imagehash
from pathlib import Path
import numpy as np


def combined_hash_int(path: Path, phash_size: int = 8, colorhash_size: int = 3) -> tuple[int, int]:
    """
    Return (combined_hash_as_int, total_bits)

    - pHash is stored in the higher bits
    - colorhash is appended in the lower bits
    """

    with Image.open(path) as im:
        im = im.convert("RGB")

        # --- pHash ---
        ph = imagehash.phash(im, hash_size=phash_size)
        ph_val = 0
        ph_bits = 0
        for row in ph.hash:
            for b in row:
                ph_val = (ph_val << 1) | int(b)
                ph_bits += 1

        # --- colorhash ---
        ch = imagehash.colorhash(im, binbits=colorhash_size)
        ch_val = 0
        ch_bits = 0
        for row in ch.hash:
            for b in row:
                ch_val = (ch_val << 1) | int(b)
                ch_bits += 1

    # --- combine ---
    combined_val = (ph_val << ch_bits) | ch_val
    total_bits = ph_bits + ch_bits

    return combined_val, total_bits


def phash(path: Path, hash_size: int):
    """Return (hash_as_int, nbits). hash_size=8 => 64 bits."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        h = imagehash.phash(im, hash_size=hash_size)

    return h


def phash_int(path: Path, hash_size: int) -> tuple[int, int]:
    """Return (hash_as_int, nbits). hash_size=8 => 64 bits."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        h = imagehash.phash(im, hash_size=hash_size)
    bits = np.asarray(h.hash, dtype=np.uint8).reshape(-1)
    nbits = int(bits.size)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val, nbits
