from transformers import AutoTokenizer, AutoModelForCausalLM # 2026.03.11. (4.57.3 -> 5.3.0)
import torch
import vars as v

# ================================================================

def line():
    print("\n", "="*64, "\n")

def setdevice():
    if torch.cuda.is_available():
        return True
    else:
        return False

def setmodel(model_name, quant_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        tie_word_embeddings=False
        )
    return model, tokenizer

def prompt(model, tokenizer, user_prompt, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
	    messages,
	    add_generation_prompt=True,
	    tokenize=True,
	    return_dict=True,
	    return_tensors="pt",
    ).to(model.device)

def response(model, tokenizer, inputs):
    outputs = model.generate(**inputs, max_new_tokens=32768)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

def summ(summ_model, summ_tockenizer, user_prompt, outputs, summ_system_prompt):
    summUserPrompt = f"The user said: {user_prompt}. {v.Who} said: {outputs}."
    summInputs = prompt(summ_model, summ_tockenizer, summUserPrompt, summ_system_prompt)
    return response(summ_model, summ_tockenizer, summInputs)