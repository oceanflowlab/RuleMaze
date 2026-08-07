import os
import yaml


def decode_base64_to_ndarray(encoded_string):
    import base64
    import cv2
    import numpy as np

    decoded_bytes = base64.b64decode(encoded_string)
    nparr = np.frombuffer(decoded_bytes, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_array


def encode_ndarray_to_base64(img_array):
    import base64
    import cv2

    _, buffer = cv2.imencode('.png', img_array)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return encoded_string


def resize_img(img, h=640, w=640):
    import cv2

    size_factor_h, size_factor_w = h / img.shape[0], w / img.shape[1]
    img_size = img.shape
    new_size = (int(img_size[1] * size_factor_w),
                int(img_size[0] * size_factor_h))
    img_ = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
    return img_


def get_setting_path(setting: str = "local") -> str:
    setting_dir = os.path.join(os.path.dirname(__file__), "../path_setting")
    candidates = [f for f in os.listdir(setting_dir) if f.endswith(".yml")]
    exact_name = f"{setting}_setting.yml"
    if exact_name in candidates:
        return os.path.join(setting_dir, exact_name)
    matched = [f for f in candidates if setting.lower() in f.lower()]
    if not matched:
        raise ValueError(
            f"No setting file matching '{setting}' in {setting_dir}. "
            f"Available: {candidates}"
        )
    if len(matched) > 1:
        raise ValueError(
            f"Ambiguous setting '{setting}' matches multiple files: {matched}. "
            f"Please be more specific."
        )
    return os.path.join(setting_dir, matched[0])


def get_config(setting: str = "local") -> dict:
    path = get_setting_path(setting)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def resolve_config_path(config: dict, key: str, base_key: str = "BASED_DIR", default=None) -> str:
    value = config.get(key, default)
    if value is None:
        return None
    value = os.path.expanduser(str(value))
    if os.path.isabs(value):
        return os.path.normpath(value)
    base = os.path.expanduser(str(config.get(base_key, "")))
    return os.path.normpath(os.path.join(base, value))


def setup_cache(CACHE_ROOT):
    os.makedirs(CACHE_ROOT, exist_ok=True)
    for sub in ["hf", "transformers", "datasets", "hub", "xdg", "tmp"]:
        os.makedirs(os.path.join(CACHE_ROOT, sub), exist_ok=True)

    os.environ["HF_HOME"] = os.path.join(CACHE_ROOT, "hf")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_ROOT, "datasets")
    os.environ["XDG_CACHE_HOME"] = os.path.join(CACHE_ROOT, "xdg")
    os.environ["TMPDIR"] = os.path.join(
        CACHE_ROOT, "tmp")     # Used by pyarrow/tmp.
