"""Persistent configuration file for vectorscope hardware/calibration defaults."""

import json
import os

CONFIG_DIR = os.path.expanduser("~/.config/vectorscope")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

SAVEABLE_KEYS = {"device", "rate", "channels", "z_delay", "z_amp", "z_gamma"}


def load_config():
    """Load config from disk. Returns empty dict if file missing."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        # Only return keys we recognise
        return {k: v for k, v in cfg.items() if k in SAVEABLE_KEYS}
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read {CONFIG_FILE}: {e}")
        return {}


def save_config(updates):
    """Merge updates into existing config and write to disk."""
    cfg = load_config()
    filtered = {k: v for k, v in updates.items() if k in SAVEABLE_KEYS}
    cfg.update(filtered)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def apply_config_defaults(parser, config, subparsers=None):
    """Apply config values as argparse defaults.

    Must be applied to each subparser individually because argparse
    subparsers maintain their own defaults namespace.
    """
    if not config:
        return
    parser.set_defaults(**config)
    if subparsers:
        for sub in subparsers.values():
            sub.set_defaults(**config)
