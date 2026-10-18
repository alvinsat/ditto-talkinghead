import cv2
import numpy as np
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


class PutBackNumpy:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            self.mask_ori_float = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.mask_ori_float = mask.astype(np.float32) / 255.0

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        result = mask_warped * frame_warped + (1 - mask_warped) * frame_rgb
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result
    

class PutBack:
    def __init__(
        self,
        mask_template_path=None,
        mask_mode="face",
    ):
        if mask_mode not in {"face", "lower_face"}:
            raise ValueError(f"Unsupported putback mask mode: {mask_mode}")
        if mask_template_path is not None:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            if mask is None:
                raise ValueError(f"Could not read putback mask: {mask_template_path}")
            mask = mask.astype(np.float32) / 255.0
            mask = mask[:, :, 0]
        elif mask_mode == "lower_face":
            mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.ellipse(mask, (256, 350), (190, 145), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=12, sigmaY=12)
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = get_mask(512, 512, 0.9, 0.9)[:, :, 0]

        self.mask_ori_float = np.ascontiguousarray(mask)
        self.result_buffer = None

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Use Cython implementation for blending
        blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)

        return self.result_buffer