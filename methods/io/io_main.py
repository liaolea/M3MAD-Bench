from methods.mad_base import MAD

FINAL_TAG_INSTRUCTION = "Provide the best possible answer. End the last line with 'Final:' followed by the concise final answer."  # noqa


class IO_Main(MAD):
    """Single-agent direct IO baseline.

    Behaviors:
      - Directly answers the query without multi-step debate structure.
      - Optionally appends a standardized 'Final:' line for downstream parsers.

    Config (configs/config_main.yaml):
      add_final_tag (bool): if true, enforce final answer tag.
      temperature (float|None): override model_temperature if provided.
    """

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        cfg = getattr(self, "method_config", {}) or {}
        self.add_final_tag = bool(cfg.get("add_final_tag", True))
        self.override_temp = cfg.get("temperature", None)

    def inference(self, sample):
        # Prepare content (text or multimodal)
        prompt, multimodal_content = self.prepare_sample_content(sample)
        
        if prompt is not None:  # Text-only case
            if self.add_final_tag and "Final:" not in prompt:
                prompt = f"{prompt}\n\n{FINAL_TAG_INSTRUCTION}"
            response = self.call_llm(prompt=prompt, temperature=self.override_temp)
        else:  # Multimodal case
            # For multimodal, we need to add the final tag instruction to the text part
            if self.add_final_tag:
                multimodal_content[0]["text"] = f"{multimodal_content[0]['text']}\n\n{FINAL_TAG_INSTRUCTION}"
            response = self.call_llm(multimodal_content=multimodal_content, temperature=self.override_temp)
        
        return {"response": response}
