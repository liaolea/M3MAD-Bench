import os
from ..mad_base import MAD
from utils.utils import compose_multimodal_input

class SelfConsistency(MAD):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.parallel_num = self.method_config["parallel_num"]
    
    def inference(self, sample):
        prompt, multimodal_content = self.prepare_sample_content(sample)
        query = prompt if prompt is not None else sample["query"]

        def call_with_mode(text):
            if multimodal_content is not None:
                mm_messages = compose_multimodal_input(text, multimodal_content)
                return self.call_llm(multimodal_content=mm_messages)
            return self.call_llm(prompt=text)
        
        agent_results = [call_with_mode(query) for _ in range(self.parallel_num)]

        final_decision_instruction = self.get_final_decision_instruction(query, agent_results)
        response = call_with_mode(final_decision_instruction)

        trace = {
            "parallel_num": self.parallel_num,
            "agent_outputs": agent_results,
            "final_decision_prompt": final_decision_instruction,
        }
        return {"response": response, "trace": trace}
    
    def get_final_decision_instruction(self, query, agent_results):
        instruction = f"[Task]:\n{query}\n\n"

        for i, result in enumerate(agent_results):
            instruction += f"[Solution {i+1}]:\n{result}\n\n"

        instruction += "Given the task and all the above solutions, reason over them carefully and provide a final answer to the task."

        return instruction
