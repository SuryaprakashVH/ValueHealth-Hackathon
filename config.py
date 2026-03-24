"""
Configuration loader for LexGuard.
Reads from .streamlit/secrets.toml for local development and Streamlit Cloud.
"""

import os
import sys
from pathlib import Path

def get_config(key: str, default: str = None) -> str:
    """
    Get configuration value from secrets.toml or environment variables.
    
    Priority order:
    1. Streamlit secrets (if running in Streamlit)
    2. .streamlit/secrets.toml file
    3. Environment variables (fallback)
    4. Default value
    
    Args:
        key: Configuration key name
        default: Default value if not found
        
    Returns:
        Configuration value or default
    """
    
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except (ImportError, Exception):
        pass
    
    # Try loading from .streamlit/secrets.toml
    try:
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # fallback for older Python
        
        secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
                if key in secrets:
                    return secrets[key]
    except (ImportError, Exception):
        pass
    
    # Fallback to environment variables
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    # Return default
    if default is not None:
        return default
    
    return None
