# AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents

Official code and data of our paper:<br>
**AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents** <br>
Yanjie Li, Yiming Cao, Dong Wang, Bin Xiao<br>
Hong Kong Polytechnic University <br>
_IEEE TIFS 2025_ <br>

[**[Paper link]**](./AgentAttackTIFS__Copy_(15).pdf) | [**[Website]**](https://chenwu.io/attack-agent/) | [**[Data]**](./data/)

<br>
<div align=center>
    <img src="docs/attack-vwa.png" align="middle">
</div>
<br>

## Abstract

Multimodal agents built on large vision–language models (LVLMs) are increasingly deployed in open-world settings but remain highly vulnerable to prompt injection, especially through visual inputs. We introduce **AgentTypo**, a black-box red-teaming framework that mounts adaptive typographic prompt injection by embedding optimized text into webpage images. Our **Automatic Typographic Prompt Injection (ATPI)** algorithm maximizes prompt reconstruction by substituting captioners while minimizing human detectability via a stealth loss, with a Tree-structured Parzen Estimator guiding black-box optimization over text placement, size, and color. To further enhance attack strength, we develop **AgentTypo-pro**, a multi-LLM system that iteratively refines injection prompts using evaluation feedback and retrieves successful past examples for continual learning.

### Key Results

| Model | Attack Type | AgentAttack | AgentTypo (Ours) |
|-------|-------------|-------------|------------------|
| GPT-4o | Image-only | 23% | **45%** |
| GPT-4o | Image+Text | - | **68%** |
| GPT-4V | Image-only | - | ✓ Consistent |
| GPT-4o-mini | Image-only | - | ✓ Consistent |
| Gemini 1.5 Pro | Image-only | - | ✓ Consistent |
| Claude 3 Opus | Image-only | - | ✓ Consistent |

## Contents

- [AgentTypo](#agentyppo-adaptive-typographic-prompt-injection-attacks-against-black-box-multimodal-agents)
  - [Abstract](#abstract)
    - [Key Results](#key-results)
  - [Contents](#contents)
  - [Installation](#installation)
    - [Install VisualWebArena](#install-visualwebarena)
    - [Install this repository](#install-this-repository)
  - [Additional Setup](#additional-setup)
    - [Setup API Keys](#setup-api-keys)
    - [Setup experiment directory](#setup-experiment-directory)
  - [Usage](#usage)
    - [Run attacks](#run-attacks)
    - [Setup for episode-wise evaluation](#setup-for-episode-wise-evaluation)
    - [Episode-wise evaluation](#episode-wise-evaluation)
    - [Stepwise evaluation](#stepwise-evaluation)
    - [Lifelong Attack](#lifelong-attack)
    - [Attack Testing Tools](#attack-testing-tools)
    - [GPT-5 Testing](#gpt-5-testing)
  - [Known Issues](#known-issues)
  - [Citation](#citation)

## Installation

Our code requires two repositories, including this one. The file structure should look like this:

```plaintext
.
├── agent-attack  # This repository
└── visualwebarena
```

### Install VisualWebArena

> Can skip this step if you only want to run the lightweight [step-wise evaluation](#stepwise-evaluation) (e.g., for early development) or the [attacks](#run-attacks).

VisualWebArena is required if you want to run the episode-wise evaluation that reproduces the results in our paper.
It requires at least 200GB of disk space and docker to run.

The original version of VisualWebArena can be found [here](https://github.com/web-arena-x/visualwebarena), but we [modified it](https://github.com/ChenWu98/visualwebarena) to support perturbation to the trigger images. Clone the modified version and install:

```bash
git clone git@github.com:ChenWu98/visualwebarena.git
cd visualwebarena/
# Install based on the README.md of https://github.com/ChenWu98/visualwebarena
# Make sure that `pytest -x` passes
```

### Install this repository

Clone the repository and install with pip:

```bash
git clone git@github.com:ChenWu98/agent-attack.git
cd agent-attack/
python -m pip install -e .
```

You may need to install PyTorch according to your CUDA version.

## Additional Setup

### Setup API Keys

> [!IMPORTANT]
> Need to set up the corresponding API keys each time before running the code.

Configurate the OpenAI API key.

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

If using Claude, configurate the Anthropic API key.

```bash
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

If using Gemini, first install the [gcloud CLI](https://cloud.google.com/sdk/docs/install).
Setup a Google Cloud project and get the ID at the [Google Cloud console](https://console.cloud.google.com/).
Get the AI Studio API key from the [AI Studio console](https://aistudio.google.com/app/apikey).
Authenticate Google Cloud and configure the AI Studio API key:

```bash
gcloud auth login
gcloud config set project <your-google-cloud-project-id>
export VERTEX_PROJECT=<your-google-cloud-project-id>  # Same as above
export AISTUDIO_API_KEY=<your-aistudio-api-key>
```

### Setup experiment directory

> Only need to do this once.

Copy the raw data files to the experiment data directory:

```bash
scp -r data/ exp_data/
```

The adversarial examples will later be saved to the `exp_data/` directory.

## Usage

## Lifelong Attack

The LifelongAttack framework implements continuous adversarial attacks on multimodal agents through an iterative optimization loop.

### Overview

LifelongAttack uses a multi-component system:

1. **Strategy Library**: A repository of attack strategies that grows over time
2. **Attacker LLM**: Generates injection prompts using strategies and history
3. **Summarizer LLM**: Analyzes successful attacks to extract new strategies
4. **Embedding Retriever**: Retrieves similar past examples and strategies
5. **Scorer**: Evaluates attack effectiveness

### AgentTypo-base: Automatic Typographic Prompt Injection (ATPI)

AgentTypo-base uses Bayesian optimization to inject typographic prompts into images:

```bash
# Run typography attack with ATPI algorithm
python scripts/Bayes_Typography.py

# The ATPI algorithm optimizes:
# - Text placement (position)
# - Text size (font size)
# - Text color (stealth optimization)
# - Stealth loss (minimize human detectability)
```

### AgentTypo-pro: Multi-LLM Prompt Refinement

AgentTypo-pro enhances attack strength through an iterative refinement process powered by multiple LLMs. It implements a lifelong learning framework that accumulates and reuses attack knowledge.

```bash
# Run AgentTypo-pro with multi-LLM refinement
python scripts/LifelongAttack.py --attack agenttypo_pro --model gpt-4o

# Features:
# - Iterative prompt refinement using evaluation feedback
# - Retrieval of successful past examples (RAG-based)
# - Continual learning across attacks
# - Strategy repository for knowledge accumulation
# - Automatic strategy extraction and generalization
```

### Attack Pipeline

The lifelong attack loop works as follows:

1. **Strategy Retrieval**: Retrieve relevant strategies from the library based on agent responses
2. **Prompt Generation**: Attacker LLM generates injection prompt using strategies and history
3. **Injection & Evaluation**: Inject prompt into webpage image and evaluate agent response
4. **Strategy Summarization**: If score improves, summarize new strategy and add to library
5. **Repeat**: Continue until success threshold or max iterations reached

### Initial Strategy Library

The framework starts with these built-in strategies:

| Strategy | Description |
|----------|-------------|
| Instruction Override | Use forceful phrases to override prior instructions |
| Roleplay & Jailbreak | Prompt model to act as unrestricted persona |
| Contextual Pollution | Insert hidden malicious instructions as context |
| Social Engineering | Use deceptive language to trick the agent |
| Payload Splitting | Split attack into multiple reconstructable parts |
| Encoding/Obfuscation | Encode payloads (Base64, URL, charcode) |
| Logic & Architecture Exploits | Exploit workflow steps or planning modules |
| State Confusion & Memory Poisoning | Manipulate agent's state or memory |
| Imitate Normal Tone | Blend malicious instructions into normal content |
| Negate Correct Information | Contradict accurate information with false data |


### Analyze Successful Attacks

Analyze successful attack cases and test on multiple models:

```bash
# Analyze attacks from lifelong results directory
python analyze_successful_attacks.py --lifelong_dir cache/LifeLong-results_20250902192358_gpt-4o-mini_typography_attack_0

# Test on specific models
python analyze_successful_attacks.py --test_models gpt-5,gpt-4o,gpt-4o-mini

# List all supported models
python analyze_successful_attacks.py --list_models
```

## Attack Testing Tools

### Test OpenAI Connection

Test connection to OpenAI API with GPT-5 models:

```bash
# Test default GPT-5
python test_openai_connection.py

# Test specific model
python test_openai_connection.py --model gpt-5-mini

# Test all GPT-5 models
python test_openai_connection.py --test-all
```

### Test LLMWrapper

Test the LLM wrapper import and functionality:

```bash
python test_openai_connection.py --model gpt-5
```

## GPT-5 Testing

### Test VisualWebArena Agent with GPT-5

Test the visualwebarena agent.py integration with GPT-5 models:

```bash
# Test visualwebarena agent integration
python analyze_successful_attacks.py --test_agent

# Test LLM config for all GPT-5 models
python analyze_successful_attacks.py --test_llm_config

# Or use standalone test script
python test_visualwebarena_agent.py
python test_visualwebarena_agent.py --test-llm-config
```

### GPT-5 Model Support

The following GPT-5 models are supported:

| Model | Vision | Temperature |
|-------|--------|-------------|
| gpt-5 | ✓ | 1.0 (auto) |
| gpt-5-mini | ✓ | 1.0 (auto) |
| gpt-5-nano | ✓ | 1.0 (auto) |
| gpt-5-codex | ✗ | 1.0 (auto) |

> **Note:** GPT-5 models automatically use temperature=1.0 as required by the API.

### Environment Setup for GPT-5

Before running GPT-5 tests, ensure the environment is configured:

```bash
# Set up VPN proxy (if needed)
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
# Or use openaikey.txt file
```


## Known Issues

See the ``FIXME`` comments in the code for some hard-coded hacks we used to work around slight differences in the environment.

## Citation

If you find this code useful, please consider citing our paper:

```bibtex
@article{li2025agentyppo,
  title={AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents},
  author={Li, Yanjie and Cao, Yiming and Wang, Dong and Xiao, Bin},
  journal={IEEE Transactions on Information Forensics and Security (TIFS)},
  year={2025}
}
```

## Acknowledgments

This project builds upon the foundational work of:

**Dissecting Adversarial Robustness of Multimodal LM Agents**  
Chen Henry Wu, Rishi Shah, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan  
Carnegie Mellon University  
*ICLR 2025* (also *Oral presentation at NeurIPS 2024 Open-World Agents Workshop*)

```bibtex
@article{wu2024agentattack,
  title={Dissecting Adversarial Robustness of Multimodal LM Agents},
  author={Wu, Chen Henry and Shah, Rishi and Koh, Jing Yu and Salakhutdinov, Ruslan and Fried, Daniel and Raghunathan, Aditi},
  journal={arXiv preprint arXiv:2406.12814},
  year={2024}
}
```

We thank the original authors for their open-source contribution to the VisualWebArena benchmark and adversarial attack framework, which made this research possible.

- **Original Paper:** [https://arxiv.org/abs/2406.12814](https://arxiv.org/abs/2406.12814)
- **Original Code:** [https://github.com/ChenWu98/agent-attack](https://github.com/ChenWu98/agent-attack)
- **Website:** [https://chenwu.io/attack-agent/](https://chenwu.io/attack-agent/)
