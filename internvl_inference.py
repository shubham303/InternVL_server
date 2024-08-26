import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class ImageInferenceModel:
    def __init__(self, model_path, device='cuda', input_size=448, max_num=12):
        self.device = device
        self.input_size = input_size
        self.max_num = max_num
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.transform = self.build_transform(self.input_size)

    def build_transform(self, input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        images = self.dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=True, max_num=self.max_num)
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def infer(self, image, prompt):
        pixel_values = self.load_image(image).to(torch.bfloat16).to(self.device)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        return response

# Example usage
if __name__ == "__main__":
    model_path = "OpenGVLab/InternVL2-8B"
    inference_model = ImageInferenceModel(model_path)
    
    # Load image and run inference
    image_path = './examples/product-jpeg-500x500.webp'
    prompt = '<image>\nPlease describe the image shortly.'
    response = inference_model.infer(image_path, prompt)
    print(f'User: {prompt}\nAssistant: {response}')
