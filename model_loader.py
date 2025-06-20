from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

class ModelLoader:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        print("Loading model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            print("Model loaded!")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def generate_response(self, input_text, chat_history_ids=None):
        if not self.model or not self.tokenizer:
            return "Model not loaded", None
        
        try:
            new_user_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
            
            if chat_history_ids is not None:
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            else:
                bot_input_ids = new_user_input_ids
            
            with torch.no_grad():
                chat_history_ids = self.model.generate(
                    bot_input_ids,
                    max_length=bot_input_ids.shape[-1] + 30,
                    num_beams=3,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones(bot_input_ids.shape, dtype=torch.long)
                )
            
            response = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            
            return response, chat_history_ids
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Sorry, I had trouble generating a response.", chat_history_ids