import os

def get_api_key(file_path="openaikey.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read().strip()
    return os.environ.get("OPENAI_API_KEY")

def get_huggingface_token(file_path="huggingfacekey.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read().strip()
    return os.environ.get("HUGGINGFACE_TOKEN")

def get_deepseek_key(file_path="deepseekkey.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read().strip()
    return os.environ.get("DEEPSEEK_API_KEY")
