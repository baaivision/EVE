from eve.model.builder import load_pretrained_model
from eve.mm_utils import get_model_name_from_path
from eve.eval.run_eve import eval_model

model_path = "Absolute Path of BAAI/EVE-7B-HD-v1.0"

prompt = "Please describle image contents in detail."
image_file = "examples/ocr_beijing.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)