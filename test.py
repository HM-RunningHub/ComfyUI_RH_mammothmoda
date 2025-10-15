import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from mammothmoda2.model import DEFAULT_NEGATIVE_PROMPT, Mammothmoda2Model
from mammothmoda2.utils import decode_diffusion_image
import time
import os
import re

def debug_gpu(name):
    output = os.popen("nvidia-smi | grep 'MiB /'").read()
    # 提取形如“400MiB / 24564MiB”的内容
    matches = re.findall(r'\d+MiB / +\d+MiB', output)
    print("\n".join(matches), name)
    time.sleep(1)

model_base = '/workspace/ComfyUI/models/MammothModa2-Preview'

# Mammothmoda2 model and processor loading.
model = Mammothmoda2Model.from_pretrained(
    model_base,
    attn_implementation="flash_attention_2",
    torch_dtype="bfloat16",
    t2i_generate=True,
)#.to("cuda")

# for name, module in model.gen_transformer.named_modules():
#     print(name)
# exit()

from optimum.quanto import freeze, qint8, quantize, quantization_map, QuantizedDiffusersModel, requantize
quantize(model, qint8)
freeze(model)
print(dict(model.named_children()).keys())
#['llm_model', 'gen_vae', 'gen_transformer', 'gen_image_condition_refiner']
model.to("cuda")
# debug_gpu('model')
# model.gen_transformer.to("cpu")
# torch.cuda.empty_cache()
# debug_gpu('model.gen_transformer')
# model.gen_vae.to("cpu")
# torch.cuda.empty_cache()
# debug_gpu('model.gen_vae')
# model.gen_image_condition_refiner.to("cpu")
# torch.cuda.empty_cache()
# debug_gpu('model.gen_image_condition_refiner')
# model.llm_model.to("cpu")
# torch.cuda.empty_cache()
# debug_gpu('model.llm_model')
# exit()

processor = AutoProcessor.from_pretrained(
    model_base,
    t2i_generate=True,
    ar_height=32,
    ar_width=32,
)

# Mammothmoda2 inputs preprocessing.
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "这张图片展示一只可爱的粉红色小猪，在非常开心的吃着薯条。四周是充满多巴胺色彩的花朵。",
            },
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f'[---> {text} ---]')
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    num_images_per_prompt=1,
    cfg_scale=7.0,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    padding=True,
    padding_side="left",
    return_tensors="pt",
    return_token_type_ids=False,  # Or generate would raise error.
).to("cuda")

print(f'[--->inputs: {inputs} ---]')

# Mammothmoda2 t2i generate.
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    generated_ids, attention_mask = model.generate(**inputs)
    diff_return_info = decode_diffusion_image(
        input_ids=inputs.input_ids,
        generated_ids=generated_ids,
        attention_mask=attention_mask,
        negative_ids=inputs.get("negative_ids", None),
        negative_mask=inputs.get("negative_mask", None),
        model=model,
        tokenizer=processor.tokenizer,
        output_dir="./mammothmoda2_t2i_release",
        num_images_per_prompt=1,
        text_guidance_scale=9.0,
        vae_scale_factor=16,
        cfg_range=(0.0, 1.0),
        num_inference_steps=2,
        height=1024,
        width=1024,
    )
    print(diff_return_info)