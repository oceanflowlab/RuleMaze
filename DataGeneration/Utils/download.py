from modelscope import snapshot_download
import argparse
import os

from Utils.utils import get_config, resolve_config_path

def download_model(model_name, model_path):
    download_dir = snapshot_download(
        model_id=model_name,
        local_dir=os.path.join(model_path, model_name)

    )
    print(f"Model downloaded to: {download_dir}")
    return download_dir

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--setting', type=str, default='local',
                    help='Settings file prefix in path_setting/*.yml')
    args.add_argument('--model_name', type=str, default=None,
                    help='Model name to download from ModelScope')
    args = args.parse_args()

    config = get_config(args.setting)
    model_name = args.model_name or config["MODEL_NAME"]["QWEN"]
    model_path = resolve_config_path(config, "MODEL_DIR")
    
    download_model(model_name, model_path)

if __name__ == "__main__":
    main()
