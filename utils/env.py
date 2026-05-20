import os


def load_dotenv_file(env_path=".env"):
    """加载简单的 .env 文件内容，并且不覆盖已有环境变量。"""
    if not env_path or not os.path.exists(env_path):
        return {}

    loaded = {}
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if key not in os.environ:
                os.environ[key] = value
            loaded[key] = os.environ[key]
    return loaded
