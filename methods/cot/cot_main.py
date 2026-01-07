from methods.mad_base import MAD

class CoT(MAD):
    def __init__(self, general_config, method_config_name=None):
        super().__init__(general_config, method_config_name)
    
    def inference(self, sample):
        # Prepare content (text or multimodal)
        prompt, multimodal_content = self.prepare_sample_content(sample)
        
        if prompt is not None:  # Text-only case
            prompt = prompt + "\n\nLet's think step by step."
            response = self.call_llm(prompt=prompt)
        else:  # Multimodal case
            # Add CoT instruction to the text part
            multimodal_content[0]["text"] = multimodal_content[0]["text"] + "\n\nLet's think step by step."
            response = self.call_llm(multimodal_content=multimodal_content)

        return {"response": response}
