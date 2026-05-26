"""配置加载工具:全局读取 config/config.yaml。"""
import os
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_root() -> str:
    return _PROJECT_ROOT


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(p: str) -> str:
    """把 config 里的相对路径解析为基于项目根目录的绝对路径。"""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_PROJECT_ROOT, p))
