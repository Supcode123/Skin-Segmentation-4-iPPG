import random
import cv2
import numpy as np
import albumentations as A
from noise import pnoise2

class RandomSkinLight(A.ImageOnlyTransform):

    """
    Generate uneven lighting (local highlights/shadows)
    """
    def __init__(self, always_apply=False, p=0.5):

        super().__init__(always_apply, p)

    def apply(self, img, mask=None, **params):
        """
        apply random lighting enhancement, only effective for skin areas
        :param img: input image (H, W, C)
        :param mask: skin area mask (H, W, 1), skin is 1, non-skin is 0
        :return: processed image

        """
        h, w, _ = img.shape

        # create a black mask the same size as the image
        light_mask = np.zeros_like(img, dtype=np.float32)

        # generate random lighting areas (circular highlight areas)
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(100, 200)
        cv2.circle(light_mask, center, radius, (1, 1, 1), -1)

        # make the edges of the lit area softer (Gaussian blur)
        light_mask = cv2.GaussianBlur(light_mask, (99, 99), 0)

        # set light intensity (alpha value affects light transparency)
        alpha = np.random.uniform(0.5, 0.9)

        skin_mask = np.isin(mask, [1, 2])
        # apply lighting regions to the skin areas of the image
        img = img.astype(np.float32)
        img[~skin_mask] = img[~skin_mask]  # non-skin areas remain unchanged
        img[skin_mask] = cv2.addWeighted(img[skin_mask], 1, light_mask[mask == 1] * 255, alpha, 0)

        return np.clip(img, 0, 255).astype(np.uint8)


class GradientLighting(A.ImageOnlyTransform):
    def __init__(self, alpha_range=(0.5, 1.5), p=0.5):
        super().__init__(always_apply=False, p=p)
        self.alpha_range = alpha_range

    def apply(self, img, **params):
        h, w, _ = img.shape
        alpha = np.linspace(self.alpha_range[0], self.alpha_range[1], w)  # horizontal gradient
        gradient = np.tile(alpha, (h, 1))
        gradient = cv2.merge([gradient, gradient, gradient])  # 3 channels
        return (img * gradient).astype(np.uint8)


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

def perlin_noise_texture(image, mask, scale=10, intensity=0.3):
    h, w, _ = image.shape
    #  Perlin noise
    noise = generate_perlin_noise(h, w, scale)
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)  # generate 3-channel noise

    # keep only the skin area (label=1,2)
    skin_mask = np.isin(mask, [1, 2]).astype(np.uint8)
    skin_mask = np.expand_dims(skin_mask, axis=-1)  # (H, W, 1)

    image_float = image.astype(np.float32)
    noise_applied = cv2.addWeighted(image, 1, noise, intensity, 0).astype(np.float32)

    enhanced_image = image_float * (1 - skin_mask) + noise_applied * skin_mask
    enhanced_image = enhanced_image.astype(np.uint8)

    return enhanced_image

# ** Perlin Noise, emulating real skin **
class PerlinTexture(A.ImageOnlyTransform):
    def __init__(self, scale=10, intensity=0.3, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.scale = scale
        self.intensity = intensity

    def apply(self, img, mask=None, **params):
        if mask is None:
            return img
        skin_mask = np.isin(mask, [1, 2]).astype(np.uint8)
        return perlin_noise_texture(img, skin_mask, self.scale, self.intensity)

# ** Randomly convert parts of the image to grayscale **
class RandomGrayScale(A.ImageOnlyTransform):
    def __init__(self, p=0.3):
        super().__init__(always_apply=False, p=p)

    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # keep RGB form, 3 channels