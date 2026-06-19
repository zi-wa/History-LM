from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import json

class HistoryLM:
    
    QuantConfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    def __init__(self, info_dir):
        with open(info_dir, 'r', encoding='utf-8') as f:
            info = json.load(f)
            
        self.few_shot_text = ""
            
        self.max_new_tokens = info["model_config"]["max_new_tokens"]
        self.model_id = info["model_config"]["model_id"]
        self.quantized = info["model_config"]["quantized"]
        self.has_few_shots = info["model_config"]["few_shots"]
        self.has_user_template = info["model_config"]["user_template"]
        self.use_streamer = info["model_config"]["use_streamer"]
        self.tie_word_embeddings = bool(info["model_config"]["tie_word_embeddings"])

        self.system_prompt = info["system_prompt"]
        self.messages = [{"role": "system", "content": self.system_prompt},]
        
        if self.has_few_shots:
            self.few_shots = info["few_shots"]
            for i, shot in enumerate(self.few_shots, 1):
                self.few_shot_text += f"Example {i}\nInput: {shot['input']}\nOutput: {shot['output']}\n\n"
                
        if self.has_user_template:
            self.user_template = info["user_template"]
                
        if self.quantized:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config = HistoryLM.QuantConfig,
                device_map="auto",
                offload_buffers= True,
                tie_word_embeddings = self.tie_word_embeddings,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                offload_buffers= True,
                tie_word_embeddings = self.tie_word_embeddings,
            )            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, clean_up_tokenization_spaces = False,)
        if self.use_streamer:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt = True, skip_special_tokens = True,)
    
    def tokenize(self):
        return self.tokenizer.apply_chat_template(
	        self.messages,
	        add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.model.device)

    def response(self, inputs):
        if self.use_streamer:
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens= self.max_new_tokens, 
                streamer= self.streamer,
                pad_token_id= self.tokenizer.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens= self.max_new_tokens, 
                pad_token_id= self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)