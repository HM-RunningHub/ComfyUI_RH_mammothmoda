import folder_paths
from optimum.quanto import freeze, qint8, quantize
import comfy.utils
from .mammothmoda2.model import DEFAULT_NEGATIVE_PROMPT, Mammothmoda2Model
from .mammothmoda2.utils import decode_diffusion_image
import os
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import torch
import numpy as np
from PIL import Image

class RunningHub_Mammothmoda_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            },
        }

    RETURN_TYPES = ("RHMammothmoda",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    CATEGORY = "Runninghub/Mammothmoda"

    def load(self, **kwargs):
        model_base = os.path.join(folder_paths.models_dir, 'MammothModa2-Preview')
        model = Mammothmoda2Model.from_pretrained(
            model_base,
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
            t2i_generate=True,
        )
        quantize(model, qint8)
        freeze(model)

        processor = AutoProcessor.from_pretrained(
            model_base,
            t2i_generate=True,
            ar_height=32,
            ar_width=32,
        )
        #['llm_model', 'gen_vae', 'gen_transformer', 'gen_image_condition_refiner']
        # model.to('cuda')
        return ({'model':model, 'processor': processor}, )

class RunningHub_Mammothmoda_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("RHMammothmoda", ),
                "prompt": ("STRING", {"multiline": True,
                                      'default': ''}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "num_inference_steps": ("INT", {"default": 50}),
                "cfg_scale": ("FLOAT", {"default": 7.0}),
                "text_guidance_scale": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 20, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "sample"
    CATEGORY = "Runninghub/Mammothmoda"

    def sample(self, **kwargs):
        width = kwargs.get('width')
        height = kwargs.get('height')
        prompt = kwargs.get('prompt')
        _model = kwargs.get('model')
        model = _model['model']
        processor = _model['processor']
        num_inference_steps = kwargs.get('num_inference_steps')
        cfg_scale = kwargs.get('cfg_scale')
        text_guidance_scale = kwargs.get('text_guidance_scale')

        self.pbar = comfy.utils.ProgressBar(num_inference_steps + 4)
        model.to('cuda')
        self.update()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            num_images_per_prompt=1,
            cfg_scale=cfg_scale,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_token_type_ids=False,  # Or generate would raise error.
        ).to("cuda")

        self.update()

        # Mammothmoda2 t2i generate.
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids, attention_mask = model.generate(**inputs)
            image = decode_diffusion_image(
                input_ids=inputs.input_ids,
                generated_ids=generated_ids,
                attention_mask=attention_mask,
                negative_ids=inputs.get("negative_ids", None),
                negative_mask=inputs.get("negative_mask", None),
                model=model,
                tokenizer=processor.tokenizer,
                output_dir="./mammothmoda2_t2i_release",
                num_images_per_prompt=1,
                text_guidance_scale=text_guidance_scale,
                vae_scale_factor=16,
                cfg_range=(0.0, 1.0),
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                update_func=self.update,
            )

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        model.to('cpu')

        return (image, )

    def update(self):
        self.pbar.update(1)

NODE_CLASS_MAPPINGS = {
    "RunningHub Mammothmoda Loader": RunningHub_Mammothmoda_Loader,
    "RunningHub Mammothmoda T2I Sampler": RunningHub_Mammothmoda_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub Mammothmoda Loader": "RunningHub Mammothmoda Loader",
    "RunningHub Mammothmoda T2I Sampler": "RunningHub Mammothmoda T2I Sampler",
} 