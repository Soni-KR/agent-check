"""Free local Ollama by default; OpenAI is an optional explicit provider."""
import os

import httpx


def provider():
    return os.getenv('LLM_PROVIDER', 'ollama').strip().lower()


def model_name():
    return (os.getenv('OLLAMA_MODEL', 'qwen2.5:7b') if provider() == 'ollama'
            else os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'))


def ollama_url():
    return os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')


def configuration_error():
    if provider() not in ('ollama', 'openai'):
        return 'LLM_PROVIDER must be ollama or openai.'
    if provider() == 'openai' and not os.getenv('OPENAI_API_KEY'):
        return 'Set OPENAI_API_KEY in backend/.env and restart the backend, or use LLM_PROVIDER=ollama for free local inference.'
    return None


def runtime_status():
    error = configuration_error()
    if not error and provider() == 'ollama':
        try:
            response = httpx.get(f'{ollama_url()}/api/tags', timeout=2)
            response.raise_for_status()
            models = response.json()['models']
            if not any(item.get('name') == model_name() for item in models):
                error = f'Model {model_name()} is not installed. Run: ollama pull {model_name()}'
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            error = 'Start Ollama locally, then refresh this page.'
    return {'provider': provider(), 'model': model_name(), 'llm_configured': error is None,
            'runtime_message': error or ('Local model ready. No API key or paid service needed.'
                                        if provider() == 'ollama' else 'OpenAI configured.')}
