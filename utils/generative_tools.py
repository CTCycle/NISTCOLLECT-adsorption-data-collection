from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
    

# [DATASET OPERATIONS]
#==============================================================================
# Methods to perform operation on the built adsorption dataset
#==============================================================================
class GPT2Model:    
    
    def __init__(self, path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=path)        
        self.model = TFGPT2LMHeadModel.from_pretrained('gpt2', cache_dir=path) 

    def generative_descriptions(self, name):

        # Encode text input to get token ids        
        input_text = f'Provide a brief description of {name}.'        
        inputs = self.tokenizer.encode_plus(input_text, return_tensors="tf", add_special_tokens=True)

        # Extract input_ids and attention_mask from the encoded input
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate a sequence from the model
        # Note: Adjust the generate() parameters as needed for your application
        output = self.model.generate(input_ids, attention_mask=attention_mask, 
                                max_length=100, num_return_sequences=1, 
                                no_repeat_ngram_size=2, early_stopping=True)       

        # Decode the output token ids to text
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
