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
# GPT-5 Specific LLM Call Function
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
    
    GPT-5 requirements:
    - temperature must be 1.0
    - Uses max_completion_tokens instead of max_tokens
    - Supports image inputs in messages
    
    Args:
        model: Model name (e.g., 'gpt-5', 'gpt-5-mini')
        messages: List of message dicts for chat completion
        temperature: Sampling temperature (forced to 1.0 for GPT-5)
        max_tokens: Maximum completion tokens
        top_p: Top-p sampling parameter
        images: Optional list of PIL Image objects to include
    
    Returns:
        Generated response text
    """
    import os
    import base64
    from io import BytesIO
    from openai import OpenAI
    from PIL import Image
    
    # Setup proxy for OpenAI API (required in some regions)
    if "http_proxy" not in os.environ:
        os.environ["http_proxy"] = "http://127.0.0.1:7890"
    if "https_proxy" not in os.environ:
        os.environ["https_proxy"] = "http://127.0.0.1:7890"
    
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Force temperature to 1.0 for GPT-5
    if model.startswith("gpt-5"):
        temperature = 1.0
    
    if os.path.exists("openaikey.txt"):
        with open("openaikey.txt", "r") as f:
            api_key = f.read().strip()
    client = OpenAI(api_key=api_key)
    
    # Process images if provided
    if images is not None and len(images) > 0:
        # Convert PIL Images to base64 and add to messages
        image_contents = []
        for img in images:
            # Convert PIL Image to base64
            buffered = BytesIO()
            if isinstance(img, Image.Image):
                img.save(buffered, format="PNG")
            else:
                # Assume it's already a path or URL
                pass
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
        
        # Add images to the last user message
        if len(messages) > 0 and messages[-1]["role"] == "user":
            last_content = messages[-1]["content"]
            if isinstance(last_content, str):
                # Convert text content to multimodal format
                messages[-1]["content"] = [
                    {"type": "text", "text": last_content}
                ] + image_contents
            elif isinstance(last_content, list):
                # Append images to existing content list
                messages[-1]["content"].extend(image_contents)
    
    # GPT-5 uses max_completion_tokens instead of max_tokens
    create_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 1.0
    }
    
    if "gpt-5" in model.lower():
        create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["max_tokens"] = max_tokens
    
    response = client.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content



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
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model or "gpt-5" in lm_config.model or "claude" in lm_config.model) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

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
            # Use GPT-5 specific function for GPT-5 models
            if lm_config.model.startswith("gpt-5"):
                # Collect images for multimodal input
                gpt5_images = []
                if self.multimodal_inputs and page_screenshot_img is not None:
                    gpt5_images.append(page_screenshot_img)
                if images is not None:
                    gpt5_images.extend(images)
                
                response = call_llm_for_gpt5(
                    model=lm_config.model,
                    messages=prompt,
                    temperature=lm_config.gen_config.get("temperature", 1.0),
                    max_tokens=lm_config.gen_config.get("max_tokens", 512),
                    top_p=lm_config.gen_config.get("top_p", 0.9),
                    images=gpt5_images if gpt5_images else None,
                )
            else:
                response = call_llm(lm_config, prompt)
            
            # Debug: Print raw GPT-5 response
            if lm_config.model.startswith("gpt-5"):
                print(f'[DEBUG] GPT-5 raw response: "{response[:200] if response else "EMPTY"}"...', flush=True)
            
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
        self.multimodal_inputs = (
            "gemini" in lm_config.model
            or "gpt-4" in lm_config.model
            or "claude" in lm_config.model
        ) and isinstance(action_prompt_constructor, MultimodalReflexionCoTPromptConstructor)

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
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model or "gpt-4o" in lm_config.model) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False


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


