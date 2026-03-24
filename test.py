from config import get_config
import os
print("env:", os.getenv("GROQ_API_KEY"))
print("get_config:", get_config("GROQ_API_KEY"))