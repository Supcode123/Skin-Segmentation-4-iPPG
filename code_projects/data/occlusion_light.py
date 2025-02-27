import random
import cv2
import numpy as np
import albumentations as A
from noise import pnoise2


class RandomSkinLight(A.DualTransform):

    """
    Generate uneven lighting (local highlights/shadows)
    """
    def __init__(self, radius_range=(50, 100), alpha_range=(0.3, 0.6), p=0.5):
        super().__init__(always_apply=False, p=p)
        self.radius_range = radius_range
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

    def apply_to_image_and_mask(self, img, mask):
        """
           apply random lighting enhancement, only effective for skin areas
           :param img: input image (H, W, C)
           :param mask: skin area mask (H, W, 1), skin is 1, non-skin is 0
           :return: processed image

       """
        if mask is None:
            raise ValueError("Mask is None. Ensure that mask is passed in the transform pipeline.")

        h, w, _ = img.shape

        # create a black mask the same size as the image
        light_mask = np.zeros_like(img, dtype=np.float32)

        # generate random lighting areas (circular highlight areas)
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(self.radius_range[0], self.radius_range[1])
        cv2.circle(light_mask, center, radius, (1, 1, 1), -1)

        # make the edges of the lit area softer (Gaussian blur)
        light_mask = cv2.GaussianBlur(light_mask, (99, 99), 0)

        # set light intensity (alpha value affects light transparency)
        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])

        skin_mask = np.isin(mask, [1, 2])
        # apply lighting regions to the skin areas of the image
        light_effect = (light_mask * 100 * alpha).astype(np.uint8)
        img = np.where(skin_mask[:, :, None], cv2.addWeighted(img, 1, light_effect, alpha, 0), img)

        return np.clip(img, 0, 255).astype(np.uint8), mask

    def apply_with_params(self, params, image, mask=None, **kwargs):

        img, mask = self.apply_to_image_and_mask(image, mask)
        return {"image": img, "mask": mask}


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


class StripeShadow(A.ImageOnlyTransform):

    def __init__(self, num_stripes=3, angle_range=(-45, 45), stripe_width=(8, 20), alpha=(0.3, 0.7), p=0.3):
        super().__init__(p)
        self.num_stripes = num_stripes  # number of stripes
        self.angle_range = angle_range  # fringe Angle Range
        self.stripe_width = stripe_width  # single stripe width range
        self.alpha = alpha  # transparency

    def apply(self, img, **params):
        h, w, _ = img.shape

        # generate a blank mask
        shadow_mask = np.zeros((h, w), dtype=np.uint8)

        angle = random.uniform(*self.angle_range)
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        for i in range(self.num_stripes):
            stripe_width = random.randint(*self.stripe_width)
            y_pos = random.randint(0, h - stripe_width)
            cv2.rectangle(shadow_mask, (0, y_pos), (w, y_pos + stripe_width), 255, -1)

        # rotating stripes
        shadow_mask = cv2.warpAffine(shadow_mask, rotation_matrix, (w, h))
        # Random transparency
        alpha = random.uniform(*self.alpha)
        # Overlay the shadow on the original image with transparency
        img_float = img.astype(np.float32)
        shadow_mask_float = shadow_mask.astype(np.float32)
        shadow_mask_float = cv2.merge([shadow_mask_float] * 3)
        # Apply the shadow mask with transparency and intensity control
        img = cv2.addWeighted(img_float, 1, shadow_mask_float / 255 * 200, alpha, 0)
        return np.clip(img, 0, 255).astype(np.uint8)


def generate_perlin_noise(h, w, scale=10):
    """ generate Perlin noise image """
    noise = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            noise[i, j] = pnoise2(i / scale, j / scale, octaves=6)

    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    return noise.astype(np.uint8)

def add_freckles(image):
    noise = np.random.normal(loc=100, scale=50, size=image.shape).astype(np.uint8)
    freckle_mask = (np.random.rand(*image.shape[:2]) > 0.9).astype(np.uint8) * 255
    freckle_mask = cv2.GaussianBlur(freckle_mask, (5,5), 0)
    freckle_layer = cv2.bitwise_and(noise, noise, mask=freckle_mask)
    return cv2.addWeighted(image, 0.9, freckle_layer, 0.1, 0)

def perlin_noise_texture_with_freckle_layer(image, mask, scale=10,
                                            intensity=0.1,freckle_intensity=0.1):
    h, w, _ = image.shape
    #  Perlin noise
    noise = generate_perlin_noise(h, w, scale)
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)  # generate 3-channel noise

    # Add freckles using Gaussian noise
    freckle_noise = np.random.normal(loc=100, scale=20, size=image.shape).astype(np.uint8)
    freckle_mask = (np.random.rand(*image.shape[:2]) > 0.95).astype(np.uint8) * 255
    freckle_mask = cv2.GaussianBlur(freckle_mask, (1, 1), 0)

    # keep only the skin area (label=1,2)
    skin_mask = np.isin(mask, [1, 2]).astype(np.uint8)
    skin_mask = np.expand_dims(skin_mask, axis=-1)  # (H, W, 1)

    # Apply freckle_mask only to the skin area
    freckle_layer = cv2.bitwise_and(freckle_noise, freckle_noise, mask=freckle_mask)
    freckle_layer = cv2.bitwise_and(freckle_layer, freckle_layer, mask=skin_mask)

    image_float = image.astype(np.float32) / 255.0  # 归一化
    noise_float = noise.astype(np.float32) / 255.0
    freckle_float = freckle_layer.astype(np.float32) / 255.0  # Normalize freckle layer

    # Apply Perlin noise and freckles to the skin region
    if np.random.rand() < 0.5:
        # Apply freckles instead of Perlin noise
        enhanced_image = image_float * (1 - skin_mask) + \
                         ( 1 - freckle_intensity) * image_float * skin_mask + \
                         freckle_intensity * freckle_float * skin_mask
    else:
        # Apply Perlin noise instead of freckles
        enhanced_image = image_float * (1 - skin_mask) + \
                         (1 - intensity) * image_float * skin_mask + \
                         intensity * noise_float * skin_mask

        # Convert back to uint8 and clip to valid range
    final_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
    return final_image


# ** Noise, emulating real skin **
class SkinTexture(A.DualTransform):
    def __init__(self, scale=5, intensity=0.06, freckle_intensity=0.1,
                  always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.scale = scale
        self.intensity = intensity
        self.freckle_intensity = freckle_intensity

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        return mask

    def apply_with_params(self, params, image, mask=None, **kwargs):

        enhanced_img = perlin_noise_texture_with_freckle_layer(image, mask,
                                                               self.scale, self.intensity,
                                                               self.freckle_intensity)
        return {"image": enhanced_img, "mask": mask}


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