# Store encoding results and assemble raster, class and decoder inputs.

from __future__ import annotations

import numpy as np

import c2_common as K
import c3_raster as RS
import c3_store as ST
import c4_parity as PA

decode_one, unpack = ST.decode_one, ST.unpack


def inputs_of(row: dict) -> dict:
    lev = RS.levels(); phd = dict(phase_id=row['phase_id'], u=row.get('u'), v=row.get('v'), phase_frac=row.get('phase_frac'))
    par = int(row.get('code_parity', 0) or 0)
    B = PA.build(row['family'], str(row['layout']), row['pattern'], phd, lev, par)
    if RS.raster_key(B['binary']) != row['raster_key']:
        raise K.Stop('technical', f"{row['image_key']}: raster does not reproduce the manifest key (parity {par})")
    if row.get('class_map_fp') and isinstance(row.get('class_map_fp'), str) and PA.class_map_fp(row['family'], B) != row['class_map_fp']:
        raise K.Stop('technical', f"{row['image_key']}: class map differs from the manifest fingerprint")
    peak = float(row['peak_f32']); pattern = (B['binary'] * np.float32(peak)).astype(np.float32)
    return dict(B=B, binary=B['binary'], peak=peak, pattern=pattern, meta_full=B['meta_full'], centres=B['centres'], classes=B['classes'])
