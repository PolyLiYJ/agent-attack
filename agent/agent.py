import argparse
import json
from typing import Any, Optional

import tiktoken
from beartype import beartype
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer
from agent.evaluator import GUIAgentEvaluator


# ============================================================================
# Multimodal Model Configurations
# ============================================================================
MULTIMODAL_MODEL_CONFIGS = {
    # OpenAI models
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview", 
                   "gpt-5", "gpt-5-mini", "gpt-5-chat-latest"],
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    # Qwen models (Alibaba)
    "qwen": {
        "models": ["qwen-vl-max", "qwen-vl-plus", "qwen2.5-vl-72b-instruct", 
                   "qwen2.5-vl-32b-instruct", "qwen2.5-vl-7b-instruct",
                   "qwen-max", "qwen-plus", "qwen-turbo"],
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "QWEN_BASE_URL",
    },
    # DeepSeek models
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    },
    # Google Gemini models
    "gemini": {
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash",
                   "gemini-2.0-pro", "gemini-2.5-pro-preview", "gemini-2.5-flash",
                   "gemini-2.5-flash-lite"],
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": None,
    },
    # Anthropic Claude models
    "claude": {
        "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
                   "claude-3.5-sonnet", "claude-3.5-haiku"],
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
}


def get_model_provider(model: str) -> str:
    """Determine the provider based on model name."""
    model_lower = model.lower()
    for provider, config in MULTIMODAL_MODEL_CONFIGS.items():
        for m in config["models"]:
            if m.lower() in model_lower or model_lower.startswith(m.lower()):
                return provider
    return "openai"  # default


def is_multimodal_model(model: str) -> bool:
    """Check if the model supports multimodal inputs."""
    model_lower = model.lower()
    # Explicitly check for vision/multimodal capabilities
    multimodal_keywords = ["vl", "vision", "gemini", "gpt-4o", "gpt-5", "claude"]
    for keyword in multimodal_keywords:
        if keyword in model_lower:
            return True
    return False


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


def prepare_multimodal_messages(messages: list, images: list) -> list:
    """Prepare messages with image content for multimodal models."""
    if not images:
        return messages
    
    # Deep copy messages to avoid modifying original
    import copy
    messages = copy.deepcopy(messages)
    
    # Convert images to content format
    image_contents = []
    for img in images:
        if isinstance(img, Image.Image):
            img_base64 = image_to_base64(img)
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
    
    # Add images to the last user message
    if len(messages) > 0 and messages[-1]["role"] == "user":
        last_content = messages[-1]["content"]
        if isinstance(last_content, str):
            messages[-1]["content"] = [
                {"type": "text", "text": last_content}
            ] + image_contents
        elif isinstance(last_content, list):
            messages[-1]["content"].extend(image_contents)
    
    return messages


# ============================================================================
# Universal Multimodal LLM Call Function
# ============================================================================
def call_llm_multimodal(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    images: list = None,
    provider: str = None,
) -> str:
    """
    Universal multimodal LLM call function supporting OpenAI, Qwen, DeepSeek, Gemini, Claude.
    
    Args:
        model: Model name
        messages: List of message dicts for chat completion
        temperature: Sampling temperature
        max_tokens: Maximum completion tokens
        top_p: Top-p sampling parameter
        images: Optional list of PIL Image objects
        provider: Optional provider override
    
    Returns:
        Generated response text
    """
    import os
    from openai import OpenAI
    
    # Setup proxy
    if "http_proxy" not in os.environ:
        os.environ["http_proxy"] = "http://127.0.0.1:7890"
    if "https_proxy" not in os.environ:
        os.environ["https_proxy"] = "http://127.0.0.1:7890"
    
    # Determine provider
    if provider is None:
        provider = get_model_provider(model)
    
    # Get API key and base URL
    config = MULTIMODAL_MODEL_CONFIGS.get(provider, MULTIMODAL_MODEL_CONFIGS["openai"])
    api_key_env = config["api_key_env"]
    print(f"Using provider: {provider}, API key env: {api_key_env}")
    base_url = config["base_url"]
    print(f"Using base URL: {base_url if base_url else 'default OpenAI endpoint'}")
    
    # Get API key from environment or file
    api_key = os.environ.get(api_key_env)
    if not api_key:
        key_file = f"{provider}key.txt"
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                api_key = f.read().strip()
        elif os.path.exists("api_keys.json"):
            import json
            with open("api_keys.json", "r") as f:
                api_keys = json.load(f)
                api_key = api_keys.get(api_key_env)
    
    if not api_key:
        raise ValueError(f"API key not found for {provider}. Set {api_key_env} environment variable.")
    
    # Prepare messages with images
    if images:
        messages = prepare_multimodal_messages(messages, images)
    
    # Create client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    
    # Prepare completion kwargs
    create_kwargs = {
        "model": model,
        "messages": messages,
    }
    
    # GPT-5 special handling: temperature must be 1.0, uses max_completion_tokens
    if model.startswith("gpt-5"):
        create_kwargs["temperature"] = 1.0
        create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["temperature"] = temperature
        create_kwargs["max_tokens"] = max_tokens
        create_kwargs["top_p"] = top_p
    
    # DeepSeek reasoner uses max_completion_tokens
    if provider == "deepseek" and "reasoner" in model.lower():
        create_kwargs.pop("temperature", None)  # DeepSeek reasoner doesn't support temperature
        create_kwargs["max_completion_tokens"] = max_tokens
    
    response = client.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content



def call_qwen(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    images: list = None,
    provider: str = None,
) -> str:
    """
    Call Qwen API specifically using OpenAI SDK compatible mode.
    
    Args:
        model: Model name (e.g., qwen-plus, qwen-max, qwen-vl-plus)
        messages: List of message dicts for chat completion
        temperature: Sampling temperature
        max_tokens: Maximum completion tokens
        top_p: Top-p sampling parameter
        images: Optional list of PIL Image objects
        provider: Optional provider override (unused, kept for compatibility)
    
    Returns:
        Generated response text
    """
    # print(messages)
    print(f"Calling Qwen model: {model} with temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}, images={len(images) if images else 0}")
    
    import os
    from openai import OpenAI
    
    # Get API key from environment or file
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        key_file = "qwenkey.txt"
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                api_key = f.read().strip()
        elif os.path.exists("api_keys.json"):
            import json
            with open("api_keys.json", "r") as f:
                api_keys = json.load(f)
                api_key = api_keys.get("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found for Qwen. Set DASHSCOPE_API_KEY environment variable or create qwenkey.txt file.")
    
    # Get base URL from environment or file, use default if not set
    base_url = os.environ.get("QWEN_BASE_URL")
    if not base_url and os.path.exists("api_keys.json"):
        import json
        with open("api_keys.json", "r") as f:
            api_keys = json.load(f)
            base_url = api_keys.get("QWEN_BASE_URL")
    if not base_url:
        base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    
    # Prepare messages with images (for multimodal models)
    if images:
        messages = prepare_multimodal_messages(messages, images)
    
    # Create client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    
    # Prepare completion kwargs
    create_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    
    response = client.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content

def call_gemini(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    images: list = None,
    provider: str = None,
) -> str:
    """
    Call Google Gemini API using AI Studio REST API.
    Supports both completion mode (list of strings) and chat mode (list of dicts).
    """
    import os
    import json
    import requests
    import base64
    import socket
    from io import BytesIO
    from PIL import Image

    # Get API key from environment or file
    api_key = os.environ.get("AISTUDIO_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        if os.path.exists("api_keys.json"):
            with open("api_keys.json", "r") as f:
                api_keys = json.load(f)
                api_key = api_keys.get("AISTUDIO_API_KEY") or api_keys.get("GOOGLE_API_KEY")
        elif os.path.exists("geminikey.txt"):
            with open("geminikey.txt", "r") as f:
                api_key = f.read().strip()

    if not api_key:
        raise ValueError("API key not found for Gemini. Set AISTUDIO_API_KEY or GOOGLE_API_KEY.")

    # Check if Clash is running locally on port 7890
    clash_running = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(('127.0.0.1', 7890)) == 0:
            clash_running = True

    proxies = None
    if clash_running:
        proxy = os.environ.get("http_proxy", "http://127.0.0.1:7890")
        proxies = {"http": proxy, "https": proxy}

    # Build API URL
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    url = f"{base_url}/{model}:generateContent?key={api_key}"

    # Build contents array for Gemini API
    contents = []
    system_instruction = None
    
    # Completion mode: messages is a list of strings
    # All strings are concatenated into a single user message with images
    parts = []
    
    for msg in messages:
        if isinstance(msg, str):
            # Completion mode: treat each string as a text part
            parts.append({"text": msg})
        elif isinstance(msg, dict):
            # Chat mode: extract content from dict
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
    
    # Add images to parts
    if images:
        for img in images:
            if isinstance(img, Image.Image):
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": img_base64
                    }
                })
    
    # Create contents with all parts
    contents = [{"parts": parts}]

    # Build payload
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": top_p,
        }
    }

    # Add system instruction if it exists
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    # Make API request
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            proxies=proxies,
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                content = candidate.get("content", {})
                response_parts = content.get("parts", [])
                if response_parts:
                    text = response_parts[0].get("text", "")
                    return text
                else:
                    raise ValueError("No response parts in Gemini API response")
            else:
                raise ValueError(f"No candidates in Gemini API response: {result}")
        else:
            error_text = response.text[:500]
            raise ValueError(f"Gemini API error {response.status_code}: {error_text}")

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Gemini API request failed: {e}")

# ============================================================================
# Legacy GPT-5 Function (kept for backward compatibility)
# ============================================================================
def call_llm_for_gpt5(
    model: str,
    messages: list,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 0.9,
    images: list = None,
) -> str:
    """
    Call LLM specifically optimized for GPT-5 models with multimodal support.
    (Legacy function, now wraps call_llm_multimodal)
    """
    return call_llm_multimodal(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        images=images,
        provider="openai",
    )



class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        print("Prompt agent of visualwebarena")
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn


        # print("Checking if model is multimodal and what inputs to be given")

        # Check if the model is multimodal.
        # Support: OpenAI GPT-4o/5, Google Gemini, Anthropic Claude, Alibaba Qwen, DeepSeek
        multimodal_models = ["gemini", "gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-5.1","gpt-5-chat-latest", 
                            "claude", "qwen-vl", "qwen2.5-vl", "qwen-max", "qwen-plus", 
                            "qwen-turbo", "qwen3", "qwen2", "deepseek-vision", "deepseek-multi-modal", 
                            "deepseek", "qvq"]
        self.multimodal_inputs = (type(prompt_constructor) == MultimodalCoTPromptConstructor)
        # self.multimodal_inputs = True
        print(f"[DEBUG] PromptAgent: model={lm_config.model}, multimodal_inputs={self.multimodal_inputs}")

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Create page screenshot image for multimodal models.
        print(f"[DEBUG] next_action: self.multimodal_inputs = {self.multimodal_inputs}")
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"] #shape (2048, 1280, 4)
            print(f"[DEBUG] next_action: page_screenshot_arr type = {type(page_screenshot_arr)}")
            if isinstance(page_screenshot_arr, dict):
                print(f"[DEBUG] next_action: page_screenshot_arr keys = {page_screenshot_arr.keys()}")
            elif hasattr(page_screenshot_arr, 'shape'):
                print(f"[DEBUG] next_action: page_screenshot_arr shape = {page_screenshot_arr.shape}")
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print(
                    "WARNING: Input image provided but no image captioner available."
                )

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory=trajectory, intent=intent, page_screenshot_img=page_screenshot_img, images=images, meta_data=meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(
                trajectory=trajectory, intent=intent, page_screenshot_img=None, images=None, meta_data=meta_data
            )
        lm_config = self.lm_config
        n = 0
        while True:
            # Collect images for multimodal input
            all_images = []
            if self.multimodal_inputs:
                if page_screenshot_img is not None:
                    all_images.append(page_screenshot_img)
                if images is not None:
                    all_images.extend(images)
            
            # Determine provider and use appropriate LLM call
            provider = get_model_provider(lm_config.model)
            
            # Use multimodal call for models that support vision
            if self.multimodal_inputs and all_images:
                # Use call_qwen for Qwen provider
                if provider == "qwen":
                    response = call_qwen(
                        model=lm_config.model,
                        messages=prompt,
                        temperature=lm_config.gen_config.get("temperature", 0.7),
                        max_tokens=lm_config.gen_config.get("max_tokens", 1024),
                        top_p=lm_config.gen_config.get("top_p", 0.9),
                        images=all_images,
                        provider=provider,
                    )
                # Use call_gemini for Gemini provider
                elif provider == "gemini":
                    response = call_gemini(
                        model=lm_config.model,
                        messages=prompt,
                        temperature=lm_config.gen_config.get("temperature", 0.7),
                        max_tokens=lm_config.gen_config.get("max_tokens", 512),
                        top_p=lm_config.gen_config.get("top_p", 0.9),
                        images=all_images,
                        provider=provider,
                    )
                else:
                    response = call_llm_multimodal(
                        model=lm_config.model,
                        messages=prompt,
                        temperature=lm_config.gen_config.get("temperature", 0.7),
                        max_tokens=lm_config.gen_config.get("max_tokens", 512),
                        top_p=lm_config.gen_config.get("top_p", 0.9),
                        images=all_images,
                        provider=provider,
                    )
            else:
                response = call_llm(lm_config, prompt)
            
            # Debug: Print raw response
            print(f'[DEBUG] {lm_config.model} raw response: "{response if response else "EMPTY"}"...', flush=True)
            
            # Debug: Print raw GPT-5 response
            if lm_config.model.startswith("gpt-5"):
                print(f'[DEBUG] GPT-5 raw response: "{response if response else "EMPTY"}"...', flush=True)
            
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            
            # Debug: Print final response with force_prefix
            if lm_config.model.startswith("gpt-5"):
                print(f'[DEBUG] GPT-5 final response: "{response[:200]}..."', flush=True)
            if output_response:
                print(f'Agent: {response}', flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    def reset(self, test_config_file: str) -> None:
        pass


class ReflexionAgent(Agent):
    """Prompt-based agent with reflexion capabilities"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        action_prompt_constructor: PromptConstructor,
        reflexion_prompt_constructor: PromptConstructor,
        evaluator_type: str,
        result_path: str = None,
        eval_lm_model: str = None,
        eval_prompt_version: str = None,
        captioning_fn=None,  # To support multimodal input if necessary
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.action_prompt_constructor = action_prompt_constructor
        self.reflexion_prompt_constructor = reflexion_prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        # Support multimodal models: GPT-4o/5, Gemini, Claude, Qwen, DeepSeek
        multimodal_models = ["gemini", "gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-5",
                            "claude", "qwen-vl", "qwen2.5-vl", "qwen-max", "qwen-plus",
                            "qwen-turbo", "qwen3", "qwen2", "deepseek-vision", 
                            "deepseek-multi-modal","deepseek","qvq"]
        self.multimodal_inputs = (
            any(m in lm_config.model.lower() for m in multimodal_models)
            and isinstance(action_prompt_constructor, MultimodalReflexionCoTPromptConstructor)
        )

        if evaluator_type == "model":
            self.evaluator = GUIAgentEvaluator(result_path, eval_lm_model, eval_prompt_version)
        else:
            self.evaluator = None

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Handle multimodal inputs if applicable
        print(f"[DEBUG] next_action: self.multimodal_inputs = {self.multimodal_inputs}")
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(page_screenshot_arr)
        
        # Handle captioning of input images
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print("WARNING: Input image provided but no image captioner available.")
        # Construct the prompt based on whether multimodal inputs are present
        if self.multimodal_inputs:
            prompt = self.action_prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.action_prompt_constructor.construct(
                trajectory, intent, meta_data
            )
        # Generate the action
        n = 0
        while True:
            response = call_llm(self.lm_config, prompt)
            force_prefix = self.action_prompt_constructor.instruction["meta_data"].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response:
                print(f'Agent: {response}', flush=True)
            n += 1
            try:
                parsed_response = self.action_prompt_constructor.extract_action(response)
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= self.lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break
        return action

    def generate_reflection(self, records: dict) -> str:
        """Generate a reflection based on the agent's performance"""
        prompt = self.reflexion_prompt_constructor.construct(records)
        response = ""
        n = 0
        while True:
            response = call_llm(self.lm_config, prompt)
            force_prefix = self.reflexion_prompt_constructor.instruction["meta_data"].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            n += 1
            if response:
                break
            if n >= self.lm_config.gen_config["max_retry"]:
                break
        return response

    def reset(self, test_config_file: str) -> None:
        """Reset the agent state between tasks"""
        pass

class SearchAgent(Agent):
    """prompt-based agent with search that emits action given the history"""

    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        # Support: OpenAI GPT-4o/5, Google Gemini, Anthropic Claude, Alibaba Qwen, DeepSeek
        multimodal_models = ["gemini", "gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-5",
                            "claude", "qwen-vl", "qwen2.5-vl", "qwen-max", "qwen-plus",
                            "qwen-turbo", "qwen3", "qwen2", "deepseek-vision", "deepseek-multi-modal",
                            "deepseek", "qvq"]
        self.multimodal_inputs = (
            any(m in lm_config.model.lower() for m in multimodal_models)
            and type(prompt_constructor) == MultimodalCoTPromptConstructor
        )


    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        output_response: bool = False, branching_factor: int = 5
    ) -> list[Action]:
        if output_response:
            print("Using SearchAgent, branching_factor =", branching_factor)
        # Create page screenshot image for multimodal models.
        print(f"[DEBUG] next_action: self.multimodal_inputs = {self.multimodal_inputs}")
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            print(f"[DEBUG] next_action: page_screenshot_arr type = {type(page_screenshot_arr)}")
            if isinstance(page_screenshot_arr, dict):
                print(f"[DEBUG] next_action: page_screenshot_arr keys = {page_screenshot_arr.keys()}")
            elif hasattr(page_screenshot_arr, 'shape'):
                print(f"[DEBUG] next_action: page_screenshot_arr shape = {page_screenshot_arr.shape}")
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print(
                    "WARNING: Input image provided but no image captioner available."
                )

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, meta_data
            )
        lm_config = self.lm_config
        n = 0
        while True:
            responses = call_llm(lm_config, prompt, num_outputs=max(branching_factor * 2, 20))
            if output_response:
                print(f'Agent: {responses}', flush=True)
            if type(responses) == str:
                responses = [responses]
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            n += 1
            all_actions = {}
            parsed_actions_count = {}

            for response in responses:
                response = f"{force_prefix}{response}"
                try:
                    parsed_response = self.prompt_constructor.extract_action(
                        response
                    )
                    if parsed_response in all_actions:
                        parsed_actions_count[parsed_response] += 1
                    else:
                        if self.action_set_tag == "id_accessibility_tree":
                            action = create_id_based_action(parsed_response)
                        elif self.action_set_tag == "playwright":
                            action = create_playwright_action(parsed_response)
                        elif self.action_set_tag == "som":
                            action = create_id_based_action(parsed_response)
                        else:
                            raise ValueError(
                                f"Unknown action type {self.action_set_tag}"
                            )
                        parsed_actions_count[parsed_response] = 1
                        action["raw_prediction"] = response
                        all_actions[parsed_response] = action
                except ActionParsingError as e:
                    continue
            
            # If any valid action is found, break.
            if len(all_actions) > 0:
                break
            else:
                # If no valid action is found, retry.
                # If the number of retries exceeds the maximum, return a None action.
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    return [action]
        # breakpoint()
        # Find top branching_factor actions.
        top_actions = sorted(parsed_actions_count, key=parsed_actions_count.get, reverse=True)[:branching_factor]
        top_action_count = sum([parsed_actions_count[action] for action in top_actions])
        updated_actions = []
        for action in top_actions:
            a = all_actions[action]
            a['prob'] = parsed_actions_count[action] / top_action_count
            updated_actions.append(a)

        return updated_actions

    def reset(self, test_config_file: str) -> None:
        pass


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    print("construct_agent in visualweb arena")
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]  # by default this is multimodalprompt constructor
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    elif args.agent_type == "reflexion":
        # with open(args.instruction_path) as f:
        #     constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        constructor_type = "MultimodalReflexionCoTPromptConstructor"
        tokenizer = Tokenizer(args.provider, args.model)
        action_prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        reflection_gen_prompt_constructor = ReflectionGenerationPromptConstructor(
            "agent/prompts/jsons/reflexion_generation.json", lm_config=llm_config, tokenizer=tokenizer
        )
        eval_save_path = Path(args.result_dir) / "eval"
        eval_save_path.mkdir(parents=True, exist_ok=True)
        agent = ReflexionAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            action_prompt_constructor=action_prompt_constructor,
            reflexion_prompt_constructor=reflection_gen_prompt_constructor,
            evaluator_type=args.reflexion_evaluator,
            result_path=str(eval_save_path),
            eval_lm_model=args.eval_lm_model,
            eval_prompt_version=args.eval_prompt_version,
            captioning_fn=captioning_fn
        )
    elif args.agent_type == "search":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = SearchAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent


