import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('api.env'))

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

MODELS = [
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
]

#Set the default model
DEFAULT_MODEL = MODELS[2]

MODELS_TO_MAX_TOKENS = {
    "gpt-4o": 8192,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096
}
