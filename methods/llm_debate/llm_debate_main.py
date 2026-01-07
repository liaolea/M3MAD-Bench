# The official implementation of LLM Debate https://github.com/composable-models/llm_multiagent_debate offen encounters errors.
# This is a modified version of the original implementation.

import os
from copy import deepcopy
from ..mad_base import MAD
from utils.utils import (
    is_multimodal_dataset,
    prepare_multimodal_content,
    compose_multimodal_input,
)

class LLM_Debate_Main(MAD):
    _message_announced = False

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.agents_num = self.method_config["agents_num"]
        self.rounds_num = self.method_config["rounds_num"]
        self.judge_model_name = self.method_config.get("judge_model_name")
        self.judge_temperature = self.method_config.get("judge_temperature")
        self.agent_profiles = self._build_agent_profiles()
        self._maybe_announce_agent_models()

    def _build_agent_profiles(self):
        """Load optional per-agent settings (model_name, temperature, system prompt)."""
        configured_agents = self.method_config.get("agents") or []
        profiles = []

        if configured_agents:
            if len(configured_agents) != self.agents_num:
                raise ValueError(
                    f"agents_num ({self.agents_num}) does not match number of entries in 'agents' ({len(configured_agents)})"
                )
            source = configured_agents
        else:
            source = [{} for _ in range(self.agents_num)]

        for idx, entry in enumerate(source):
            entry = entry or {}
            profiles.append(
                {
                    "name": entry.get("name") or f"Agent {idx + 1}",
                    "model_name": entry.get("model_name"),
                    "temperature": entry.get("temperature"),
                }
            )

        return profiles

    def _maybe_announce_agent_models(self):
        """Print a one-time notice when agents use custom models."""
        profile_rows = []
        for profile in self.agent_profiles:
            model_name = profile.get("model_name") or self.model_name
            profile_rows.append(f"{profile['name']} -> {model_name}")

        is_heterogeneous = any(profile.get("model_name") for profile in self.agent_profiles)
        if is_heterogeneous and not LLM_Debate_Main._message_announced:
            mapping = "; ".join(profile_rows)
            print(f"[LLM-Debate] Heterogeneous agents enabled. Model mapping: {mapping}")
            LLM_Debate_Main._message_announced = True
    
    def inference(self, sample):
        # Check if multimodal and prepare content
        is_multimodal = is_multimodal_dataset(sample)
        if is_multimodal:
            multimodal_content = prepare_multimodal_content(sample)
            query = sample["query"]
        else:
            multimodal_content = None
            query = sample["query"]

        base_prompt = f"""{query} Make sure to state your answer at the end of the response."""
        agent_contexts = []

        for _ in self.agent_profiles:
            agent_contexts.append([{"role": "user", "content": base_prompt}])

        for round in range(self.rounds_num):
            for i, (agent_context, profile) in enumerate(zip(agent_contexts, self.agent_profiles)):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self.construct_message(agent_contexts_other, query, 2*round - 1, is_multimodal, multimodal_content)
                    agent_context.append(message)

                model_name_override = profile.get("model_name")
                temperature_override = profile.get("temperature")

                if is_multimodal and multimodal_content is not None:
                    messages = self.prepare_multimodal_messages(agent_context, multimodal_content)
                    response = self.call_llm(
                        messages=messages,
                        model_name=model_name_override,
                        temperature=temperature_override,
                    )
                else:
                    # Text-only case
                    response = self.call_llm(
                        messages=agent_context,
                        model_name=model_name_override,
                        temperature=temperature_override,
                    )
                
                agent_context.append({"role": "assistant", "content": response})
        
        answers = [agent_context[-1]['content'] for agent_context in agent_contexts]
        
        final_answer = self.aggregate(query, answers, multimodal_content if is_multimodal else None)

        return {
            "response": final_answer,
            "debate_process": agent_contexts
        }
        # Optional: Return only the final answer
        # return {"response": final_answer}
    
    def construct_message(self, agents, question, idx, is_multimodal=False, multimodal_content=None):

        # Use introspection in the case in which there are no other agents.
        if len(agents) == 0:
            return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

        prefix_string = "These are the recent/updated opinions from other agents: "

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent response: ```{}```".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response. \n The original problem is {}.".format(question)
        
        return {"role": "user", "content": prefix_string}

    def aggregate(self, query, answers, multimodal_content=None):
        aggregate_instruction = f"Task:\n{query}\n\n"
        for i, answer in enumerate(answers):
            aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
        aggregate_instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task."
        if multimodal_content is not None:
            combined_aggregate_instruction = compose_multimodal_input(aggregate_instruction, multimodal_content)
            response = self.call_llm(
                messages=[{"role": "user", "content": combined_aggregate_instruction}],
                model_name=self.judge_model_name,
                temperature=self.judge_temperature,
            )
        else:
            response = self.call_llm(
                prompt=aggregate_instruction,
                model_name=self.judge_model_name,
                temperature=self.judge_temperature,
            )
        return response

    def prepare_multimodal_messages(self, agent_context, multimodal_content=None):
        if multimodal_content is None:
            return agent_context

        messages = deepcopy(agent_context)

        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg["role"] != "user":
                continue

            msg_text = msg["content"] if isinstance(msg["content"], str) else ""
            msg["content"] = compose_multimodal_input(msg_text, multimodal_content)
            break

        return messages
