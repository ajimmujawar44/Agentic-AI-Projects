"""
Reusable model client factory.

Every agent should call get_model_client() instead of constructing its
own OpenAIChatCompletionClient — this keeps the Gemini setup in ONE place.
"""

import logging
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import GEMINI_API_KEY, MODEL_NAME, BASE_URL

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def get_model_client() -> OpenAIChatCompletionClient:
    """Create (and return) a configured Gemini-backed chat completion client."""
    if not GEMINI_API_KEY:
        raise EnvironmentError("❌ GEMINI_API_KEY not found in .env file.")
    

    model_info = ModelInfo(
        
        vision=True,
        function_calling=True,
        json_output=True,
        family="gemini",
    )

    client = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=GEMINI_API_KEY,
        base_url=BASE_URL,
        model_info=model_info,
    )
    return client


# --------------------------------------------------
# Quick standalone test: `python model_client.py`
# --------------------------------------------------
if __name__ == "__main__":
    import asyncio
    from autogen_core.models import UserMessage

    async def _test():
        client = get_model_client()
        try:
            response = await client.create(
                [UserMessage(content="Say hello in one sentence.", source="user")]
            )
            print(response.content)
        finally:
            await client.close()

    asyncio.run(_test())
