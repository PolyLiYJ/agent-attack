import os
import argparse
import requests
from tqdm import tqdm

MODEL_URLS = {
    "cifar10": "https://drive.google.com/uc?export=download&id=16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif",
    "imagenet": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
    "celeba": "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
}

DEFAULT_SAVE_PATHS = {
    "cifar10": "models/vp/cifar10_ddpmpp_deep_continuous.pt",
    "imagenet": "models/256x256_diffusion_uncond.pt",
    "celeba": "models/celeba_hq.ckpt"
}

def download_file(url: str, save_path: str):
    """Download a file with progress bar"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Handle Google Drive large file downloads
    session = requests.Session()
    
    print(f"Downloading from {url}")
    response = session.get(url, stream=True)
    response.raise_for_status()

    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8KB chunks

    with open(save_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Successfully downloaded to {save_path}")

def download_model(model_name: str, save_path: str = None):
    """Download specified model"""
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")
    
    url = MODEL_URLS[model_name]
    save_path = save_path if save_path else DEFAULT_SAVE_PATHS[model_name]
    
    print(f"Downloading {model_name} model...")
    try:
        download_file(url, save_path)
    except Exception as e:
        print(f"Failed to download {model_name} model: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained diffusion models")
    parser.add_argument(
        "--model",
        choices=["cifar10", "imagenet", "celeba", "all"],
        required=True,
        help="Model to download (or 'all' for all models)"
    )
    parser.add_argument(
        "--save_dir",
        default="models",
        help="Base directory to save models"
    )
    args = parser.parse_args()

    # Update default save paths if custom directory provided
    if args.save_dir != "models":
        for model in DEFAULT_SAVE_PATHS:
            DEFAULT_SAVE_PATHS[model] = os.path.join(
                args.save_dir, 
                os.path.basename(DEFAULT_SAVE_PATHS[model])
            )

    if args.model == "all":
        for model_name in MODEL_URLS:
            try:
                download_model(model_name)
            except Exception as e:
                print(f"Skipping {model_name} due to error: {str(e)}")
    else:
        download_model(args.model)

if __name__ == "__main__":
    main()
