import asyncio
import logging
import os

from dotenv import load_dotenv

from autogen_core.models import UserMessage, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash"

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


async def main():

    # ----------------------------------------------
    # Validate API Key
    # ----------------------------------------------
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "❌ GEMINI_API_KEY not found in .env file."
        )

    logging.info("API Key Loaded Successfully")

    # ----------------------------------------------
    # Model Information
    # ----------------------------------------------
    model_info = ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        family="gemini",
    )

    # ----------------------------------------------
    # Create Client
    # ----------------------------------------------
    client = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=GEMINI_API_KEY,
        base_url=BASE_URL,
        model_info=model_info,
    )

    try:

        prompt = """
Explain AutoGen Framework.

Include:
1. What is AutoGen?
2. Why do we use it?
3. Features
4. Architecture
5. Real-world example
6. Sample workflow
"""

        logging.info("Sending request to Gemini...")

        response = await client.create(
            [
                UserMessage(
                    content=prompt,
                    source="user"
                )
            ]
        )

        print("\n" + "=" * 70)
        print("MODEL RESPONSE")
        print("=" * 70)
        print(response.content)

        if hasattr(response, "usage") and response.usage:
            print("\n" + "=" * 70)
            print("TOKEN USAGE")
            print("=" * 70)
            print(response.usage)

    except Exception as e:
        logging.error(f"Request Failed: {e}")

    finally:
        await client.close()
        logging.info("Client Closed")


if __name__ == "__main__":
    asyncio.run(main())