import random
import cv2
import numpy as np
import albumentations as A


class Wrinkles(A.DualTransform):
    """
    Generate fine wrinkles using Canny edge detection on skin regions.
    """
    def __init__(self,contrast_factor=(0.8,1.2), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.contrast_factor = contrast_factor

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):

        return mask

    def apply_to_image_and_mask(self, img, mask):
        """
        Apply wrinkle effect only on skin areas.
        :param img: input image (H, W, C)
        :param mask: skin area mask (H, W), skin pixels > 0
        :return: processed image
        """
        if mask is None:
            raise ValueError("Mask is None. Ensure that mask is passed in the transform pipeline.")

        mask_skin = np.logical_or(mask == 1, mask == 2).astype(np.uint8)  # (H, W)
        contrast = random.uniform(self.contrast_factor[0], self.contrast_factor[1])
        # Calculate wrinkles (based on Canny edge detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 10, 50).astype(np.uint8)

        enhanced_image = img.astype(np.float32)
        enhanced_image[(edges == 255) & (mask_skin == 1)] *= contrast
        enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
        return enhanced_image

    def apply_with_params(self, params, image, mask=None, **kwargs):
        image = self.apply_to_image_and_mask(image, mask)
        return {"image": image, "mask": mask}

    def get_transform_init_args_names(self):
        return ("contrast_factor")

class GradientLighting(A.DualTransform):
    def __init__(self, alpha_range=(0.5, 1.5), p=0.5):
        super().__init__(always_apply=False, p=p)
        self.alpha_range = alpha_range

    def apply(self, img, **params):
        """
        Apply random lighting enhancement, only effective for skin areas.
        :param img: input image (H, W, C)
        :return: processed image
        """
        return img

    def apply_to_mask(self, mask, **params):
        return mask

    def apply_to_image_and_mask(self, img, mask, **kwargs):
        h, w, _ = img.shape
        alpha = np.linspace(self.alpha_range[0], self.alpha_range[1], w)  # horizontal gradient
        gradient = np.tile(alpha, (h, 1))
        gradient = cv2.merge([gradient, gradient, gradient])  # 3 channels
        mask_skin = (mask == 1) | (mask == 2)
        mask_skin = np.expand_dims(mask_skin, axis=-1)
        img = img.astype(np.float32)
        gradient = gradient.astype(np.float32)
        img = np.where(mask_skin, img * gradient, img)
        return img.astype(np.uint8), mask

    def apply_with_params(self, params, image, mask=None, **kwargs):
        img, mask = self.apply_to_image_and_mask(image, mask)
        return {"image": img, "mask": mask}


# ** Noise, emulating real skin **



class SkinColorJitter(A.DualTransform):
    def __init__(self, brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                 saturation=(0.8, 1.2),  p=0.7):
        super().__init__(p=p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        return mask

    def apply_with_params(self, params, image, mask=None, **kwargs):
        # Apply ColorJitter only on skin regions
        skin_mask = np.isin(mask, [1, 2])  # 1 and 2 are the skin and nose classes (example)

        # Apply ColorJitter to the whole image
        color_jitter = A.ColorJitter(brightness=self.brightness, contrast=self.contrast,
                                     saturation=self.saturation, p=self.p)
        enhanced_image = color_jitter(image=image)["image"]

        # Only apply the jitter to skin regions
        enhanced_image[~skin_mask] = image[~skin_mask]  # Keep original in non-skin areas

        return {"image":enhanced_image , "mask": mask}

# ** Randomly convert parts of the image to grayscale **
class RandomGrayScale(A.ImageOnlyTransform):
    def __init__(self, p=0.3):
        super().__init__(always_apply=False, p=p)

    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # keep RGB form, 3 channels