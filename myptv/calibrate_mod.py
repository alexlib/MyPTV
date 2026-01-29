# -*- coding: utf-8 -*-
"""Compatibility calibration facade.

The historical test suite expects `from myptv.calibrate_mod import calibrate`.
Internally, MyPTV supports multiple camera models with model-specific
calibrators.
"""

from myptv.TsaiModel.calibrate import calibrate_Tsai
from myptv.extendedZolof.calibrate import calibrate_extendedZolof


class calibrate:
    def __init__(self, camera, lab_coords, img_coords, **kwargs):
        # Prefer the underlying model camera if a wrapper/legacy wrapper is provided.
        model_camera = getattr(camera, "camera", None) or getattr(camera, "_cam", None) or camera
        model_name = getattr(camera, "modelName", None)

        if model_name == "extendedZolof":
            self._impl = calibrate_extendedZolof(model_camera, lab_coords, img_coords, **kwargs)
        else:
            # Default to Tsai to preserve legacy expectations.
            self._impl = calibrate_Tsai(model_camera, lab_coords, img_coords, **kwargs)

    def searchCalibration(self, *args, **kwargs):
        return self._impl.searchCalibration(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)
