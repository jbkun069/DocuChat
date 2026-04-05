from optimum.onnxruntime import ORTModelForFeatureExtraction # type: ignore
from transformers import AutoTokenizer
from pathlib import Path

model_id = "sentence-transformers/all-MiniLM-L6-v2"
save_directory = Path("onnx_model")

model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"✅ Optimized model saved to the '{save_directory}' folder!")