from transformers import GPTNeoForCausalLM, AutoTokenizer

model_name = "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)


def generate_text(prompt, max_length=50, temperature=0.9):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_length=max_length, 
        temperature=temperature, 
        do_sample=True,
        max_time=10.0
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)





