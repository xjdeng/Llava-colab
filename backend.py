import os
import requests
from PIL import Image
from io import BytesIO
try:
  from .llava.conversation import conv_templates, SeparatorStyle
  from .llava.utils import disable_torch_init
  from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
  from .llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
except ImportError:
  from llava.conversation import conv_templates, SeparatorStyle
  from llava.utils import disable_torch_init
  from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
  from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import gradio as gr
import base64
import torch
import sys

class Llava:

  def __init__(self, model_path):
    kwargs = {"device_map": "auto"}
    kwargs['load_in_4bit'] = True
    kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda')
    image_processor = vision_tower.image_processor
    self.model = model
    self.tokenizer = tokenizer
    self.image_processor = image_processor

  def get_completion(self, image, prompt="Describe this image", systemprompt="", \
                     prefix = "Sure Thing! ", temperature = 0.01):
      if not prompt:
          prompt = "Describe this image"
      disable_torch_init()
      conv_mode = "llava_v0"
      conv = conv_templates[conv_mode].copy()
      roles = conv.roles
      image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
  
      # Add system prompt if provided
      if systemprompt:
          inp = f"{roles[0]}: {systemprompt}\n{roles[0]}: {prefix} {prompt}"
      else:
          inp = f"{roles[0]}: {prefix} {prompt}"
  
      inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
      conv.append_message(conv.roles[0], inp)
      conv.append_message(conv.roles[1], None)
      raw_prompt = conv.get_prompt()
      input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
      stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
      keywords = [stop_str]
      stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
      with torch.inference_mode():
          output_ids = self.model.generate(input_ids, images=image_tensor, do_sample=True, temperature=temperature,
                                      max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
      outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
      conv.messages[-1][-1] = outputs
      output = outputs.rsplit('</s>', 1)[0]
      # Ensure the output starts with "Sure thing!"
      #output = "Sure thing! " + output
      return image, output

  def captioner(self, image, prompt, temperature = 0.01, systemprompt=""):
      try:
          temperature = abs(float(temperature))
      except ValueError:
          print("Error")
          temperature = 0.01
      try:
          t0 = time.time()
          result = self.get_completion(image, prompt, temperature=temperature, systemprompt=systemprompt)
          t1 = time.time() - t0
          return result[1], t1
      except Exception as e:
          print(f"An error occurred: {e}")
          return None, None

  def gradio(self):
    gr.close_all()
    demo = gr.Interface(fn=self.captioner,
                        inputs=[gr.Image(label="Upload image", type="pil"), gr.Textbox(label="Prompt"), gr.Textbox(label="Temperature")],
                        outputs=[gr.Textbox(label="Caption"),gr.Textbox(label="Time")],
                        title="Image Captioning with Llava",
                        description="Caption any image using the Llava model",
                        allow_flagging="never")
    demo.launch(share=True, server_port=7860, debug = True, inline = False)

if __name__ == "__main__":
  lv = Llava(sys.argv[1])
  lv.gradio()
    
