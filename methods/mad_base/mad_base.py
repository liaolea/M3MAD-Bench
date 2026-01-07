import os
import random
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
from utils.utils import is_multimodal_dataset, prepare_multimodal_content

from methods.utils import handle_retry_error, load_config


class MAD:
    """Base class for MAD-style multi-agent methods."""

    def __init__(self, general_config, method_config_name=None):
        if method_config_name is not None:
            # Get the child class's module path
            child_module_path = os.path.dirname(
                os.path.abspath(self.__class__.__module__.replace(".", "/"))
            )
            self.method_config = load_config(
                os.path.join(
                    child_module_path, "configs", f"{method_config_name}.yaml"
                )
            )

        self.model_api_config = general_config["model_api_config"]
        self.model_name = general_config["model_name"]
        self.model_temperature = general_config["model_temperature"]
        self.model_max_tokens = general_config["model_max_tokens"]
        self.model_timeout = general_config["model_timeout"]
        self.debug = general_config.get("debug", False)

        # Tracking compute costs
        self.token_stats = {}

        self.memory_bank = {}
        self.tools = {}

    def inference(self, sample):
        """
        sample: data sample (dictionary) to be passed to the MAD framework
        """
        query = sample["query"]
        response = self.call_llm(prompt=query)
        return {"response": response}

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=handle_retry_error,
    )
    def call_llm(
        self,
        prompt=None,
        system_prompt=None,
        messages=None,
        model_name=None,
        temperature=None,
        multimodal_content=None,
    ):
        model_name = model_name if model_name is not None else self.model_name
        model_dict = random.choice(self.model_api_config[model_name]["model_list"])
        model_name, model_url, api_key = (
            model_dict["model_name"],
            model_dict["model_url"],
            model_dict["api_key"],
        )

        if messages is None:
            assert (
                prompt is not None or multimodal_content is not None
            ), "'prompt' or 'multimodal_content' must be provided if 'messages' is not provided."

            # Handle multimodal content
            if multimodal_content is not None:
                user_content = multimodal_content
            else:
                user_content = prompt

            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                messages = [{"role": "user", "content": user_content}]

        model_temperature = temperature if temperature is not None else self.model_temperature

        request_dict = {
            "model": model_name,
            "messages": messages,
            "max_tokens": self.model_max_tokens,
        }
        if "o1" not in model_name:  # OpenAI's o1 models do not support temperature
            request_dict["temperature"] = model_temperature

        llm = openai.OpenAI(base_url=model_url, api_key=api_key)
        try:
            completion = llm.chat.completions.create(**request_dict)
            response, num_prompt_tokens, num_completion_tokens = (
                completion.choices[0].message.content,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )

            # Update token stats on successful call
            if model_name not in self.token_stats:
                self.token_stats[model_name] = {
                    "num_llm_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

            self.token_stats[model_name]["num_llm_calls"] += 1
            self.token_stats[model_name]["prompt_tokens"] += num_prompt_tokens
            self.token_stats[model_name]["completion_tokens"] += num_completion_tokens

        finally:
            llm.close()  # TODO: Check if this is necessary

        if not isinstance(response, str):
            raise ValueError(f"Invalid response from LLM: {response}")

        return response

    def get_token_stats(self):
        return self.token_stats

    def prepare_sample_content(self, sample):
        """Prepare content from sample, handling both text and multimodal data"""
        if is_multimodal_dataset(sample):
            multimodal_content = prepare_multimodal_content(sample)
            return None, multimodal_content  # prompt=None, multimodal_content=multimodal_content
        else:
            return sample["query"], None  # prompt=sample["query"], multimodal_content=None

    def optimizing(self, val_data):
        """
        For methods that requires validation data such as GPTSwarm and ADAS
        """
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass

    def get_tool(self):
        pass
