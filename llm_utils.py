import os
import time
import random
import sys
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()

def get_keys(prefix: str) -> List[str]:
    """Get all environment variables starting with a prefix."""
    return [v for k, v in os.environ.items() if k.startswith(prefix) and v]

def call_model_with_key_rotation(models_to_try: List[tuple], messages: List[Any], temperature: float = 0.3, max_tokens: int = 4096) -> Any:
    """
    Attempts to call an LLM, aggressively falling back through models and API keys if a 429 occurs.
    """
    from langchain_groq import ChatGroq
    from langchain_google_genai import ChatGoogleGenerativeAI
    import groq

    last_err = None
    
    def invoke_with_retry(llm, msgs, retries=1):
        for i in range(retries):
            try:
                return llm.invoke(msgs)
            except groq.RateLimitError as e:
                print(f"[Info] API rate limit/quota hit. Sleeping {(i+1)*20}s...", file=sys.stderr)
                time.sleep((i+1)*20)
                if i == retries - 1:
                    raise e
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print(f"[Info] API rate limit/quota hit. Sleeping {(i+1)*20}s...", file=sys.stderr)
                    time.sleep((i+1)*20)
                    if i == retries - 1:
                        raise e
                else:
                    raise e
    
    for provider, model_name in models_to_try:
        if provider == "groq":
            keys = get_keys("GROQ_API_KEY")
            random.shuffle(keys)
            for key in keys:
                try:
                    kwargs = {
                        "model": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "api_key": key
                    }
                    if "qwen" in model_name.lower():
                        kwargs["reasoning_format"] = "hidden"
                        
                    llm = ChatGroq(**kwargs)
                    return invoke_with_retry(llm, messages)
                except Exception as e:
                    last_err = e
                    print(f"[Warning] Groq ({model_name}) failed with key {key[:10]}... : {e}", file=sys.stderr)
                    
        elif provider == "gemini":
            keys = get_keys("GEMINI_API_KEY")
            if not keys and os.getenv("GOOGLE_API_KEY"):
                keys = [os.getenv("GOOGLE_API_KEY")]
            keys = [k for k in keys if k]
            random.shuffle(keys)
            
            for key in keys:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        google_api_key=key
                    )
                    return invoke_with_retry(llm, messages)
                except Exception as e:
                    last_err = e
                    print(f"[Warning] Gemini ({model_name}) failed with key {key[:10]}... : {e}", file=sys.stderr)
                    
    print(f"[Error] All LLM fallbacks failed. Last error: {last_err}", file=sys.stderr)
    raise ValueError("All providers and keys failed.")

def _extract_json_block(text: Any) -> dict:
    """Safely extracts JSON from an LLM response block."""
    import json
    if not isinstance(text, str):
        text = text.content if hasattr(text, 'content') else str(text)
        
    start_idx = text.find("```json")
    if start_idx != -1:
        start_idx += 7
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            text = text[start_idx:end_idx].strip()
    else:
        # Try raw block
        start_idx = text.find("```")
        if start_idx != -1:
            start_idx += 3
            end_idx = text.find("```", start_idx)
            if end_idx != -1:
                text = text[start_idx:end_idx].strip()
                
    try:
        return json.loads(text.strip())
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return {}

def _extract_python_block(text: Any) -> str:
    """Safely extracts Python code from an LLM response block."""
    import re
    if not isinstance(text, str):
        text = text.content if hasattr(text, 'content') else str(text)
        
    # Strip reasoning tags (just in case)
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
    
    start_idx = text.find("```python")
    if start_idx != -1:
        start_idx += 9
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx].strip()
            
    # Fallback
    start_idx = text.find("```")
    if start_idx != -1:
        start_idx += 3
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx].strip()
            
    return text.strip()
