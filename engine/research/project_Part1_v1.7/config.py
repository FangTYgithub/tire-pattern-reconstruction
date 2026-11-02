# Provide standalone model defaults and output paths. Study adapters supply the effective experiment
# configuration.

import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_DATA     = PROJECT_ROOT / "outputs" / "data"


SOURCE_GUTHRIE_2017 = (
    "Guthrie, A.G., Botha, T.R., Els, P.S. (2017). "
    "3D contact patch measurement inside rolling tyres. "
    "Journal of Terramechanics, 69, 13–21."
)
SOURCE_MAPPES_2023 = (
    "Mappes, V. et al. (2023). Development of a tire sensor to determine "
    "the contact patch while driving. "
    "14th Int. Munich Chassis Symposium, Springer."
)
SOURCE_WILLOMITZER_2019 = (
    "Willomitzer, F. (2019). Single-Shot 3D Sensing Close to Physical Limits "
    "and Information Limits. PhD thesis, FAU Erlangen-Nürnberg."
)
SOURCE_HZ_2003 = (
    "Hartley, R., Zisserman, A. (2003). "
    "Multiple View Geometry in Computer Vision. Cambridge University Press."
)


REF_TYRE_RADIUS_MM        = 401.0
REF_M10_OBSTACLE_MM       = 10.0
REF_MAX_DEFORMATION_MM    = 16.0
REF_INFLATION_PRESSURE_BAR= 2.0


M1_BIN_WIDTH_MM        = 1.0
M1_GRID_STEP_Y_MM      = 1.0
M1_INTERP_KIND         = "pchip"
M1_SMOOTHING_WINDOW    = 21
M1_SMOOTHING_POLYORDER = 3
M1_SURFACE_LENGTH_X_MM = 200.0
M1_GRID_STEP_X_MM      = 2.0


M1_INDENT_DEPTHS_MM    = [2.0, 5.0, 10.0]
M1_INDENT_WIDTH_MM     = 20.0


M2_IMAGE_SIZE_PX       = 512

M2_LINE_SPACING_MM     = 2.0
M2_LINE_WIDTH_MM       = 0.3

M2_DOT_SPACING_MM      = 2.0
M2_DOT_RADIUS_MM       = 0.15

M2_UNIQUENESS_LAMBDA        = 0.5
M2_UNIQUENESS_WINDOW_MM     = 5.0


M3_IMAGE_WIDTH_PX    = 512
M3_IMAGE_HEIGHT_PX   = 512
M3_FOCAL_LENGTH_MM   = 25.0
M3_PIXEL_PITCH_MM    = 0.030


M3_BASELINE_MM       = 50.0
M3_WORKING_DIST_MM   = 150.0
M3_NOISE_SIGMA       = 0.0
M3_BRIGHTNESS        = 255.0
M3_BATCH_SIZE        = 256
M3_INTENSITY_THRESH  = 0.01


M3_DEPOSIT           = "sum"
M3_PSF_SIGMA_PX      = 1.0
M3_INTERSECTOR       = "heightfield"


M3_FX_PX             = M3_FOCAL_LENGTH_MM / M3_PIXEL_PITCH_MM
M3_CX_PX             = (M3_IMAGE_WIDTH_PX  - 1) / 2.0
M3_CY_PX             = (M3_IMAGE_HEIGHT_PX - 1) / 2.0
M3_EFFECTIVE_THETA_DEG = float(
    np.degrees(2 * np.arctan(M3_BASELINE_MM / 2 / M3_WORKING_DIST_MM))
)


M4_SEARCH_MODE            = "volume"
M4_MEASUREMENT_VOLUME_MM  = 15.0

M4_ONE_TO_ONE             = True
M4_BLOB_AREA_CHECK        = True
M4_BLOB_AREA_LO           = 0.5
M4_BLOB_AREA_HI           = 1.5
M4_BLOB_ELONG_MAX         = 2.5
M5_OUTLIER_ERR_MM         = 1.0
M4_BINARY_THRESHOLD  = 0
M4_MIN_AREA_PX       = 1
M4_MAX_AREA_PX       = 5000
M4_MIN_INTENSITY     = 10


M5_UNIQUENESS_WINDOW_PX  = 32
M5_UNIQUENESS_STRIDE_PX  = 16
M5_UNIQUENESS_N_SAMPLES  = 200

M5_SWEEP_SPACING_MM      = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
M5_SWEEP_PATTERN_TYPES   = ["dots"]


M5_ETA_DELTA_Z_MODE        = "design_unique_depth"
M5_NN_DECODER_RANGE_FACTOR = 0.5


SENSOR_CONFIG = {
    "image_width_px":   M3_IMAGE_WIDTH_PX,
    "image_height_px":  M3_IMAGE_HEIGHT_PX,
    "focal_length_mm":  M3_FOCAL_LENGTH_MM,
    "pixel_pitch_mm":   M3_PIXEL_PITCH_MM,
    "baseline_mm":      M3_BASELINE_MM,
    "working_dist_mm":  M3_WORKING_DIST_MM,
    "noise_sigma":      M3_NOISE_SIGMA,
    "brightness":       M3_BRIGHTNESS,
    "batch_size":       M3_BATCH_SIZE,
    "intensity_thresh": M3_INTENSITY_THRESH,
    "deposit":          M3_DEPOSIT,
    "psf_sigma_px":     M3_PSF_SIGMA_PX,
}
