import openai # Changed from openai to openai.AsyncOpenAI below
import sys
from typing import AsyncGenerator # For async streaming
from loguru import logger # Import loguru

class LLMClientError(Exception):
    """Custom exception for LLM client errors."""
    pass

class LLMClient:
    """
    A client for interacting with an OpenAI-compatible LLM API asynchronously.
    Handles API configuration, request execution, and streaming responses.
    """
    def __init__(self, config: dict):
        """
        Initializes the LLMClient.

        Args:
            config: A dictionary containing the API configuration, typically loaded
                    from a config file, expected to have an 'openai' section with
                    'base_url', 'model', 'api_key' (optional), 'temperature', 'top_p',
                    and 'max_tokens'.
        """
        openai_config = config.get('openai')
        if not openai_config:
            raise LLMClientError("Missing 'openai' configuration section.")

        self.base_url = openai_config.get('base_url')
        self.model = openai_config.get('model')
        self.api_key = openai_config.get('api_key') # Can be None if auth is handled differently
        self.temperature = openai_config.get('temperature', 0.7)
        self.top_p = openai_config.get('top_p', 1.0)
        self.max_tokens = openai_config.get('max_tokens', 4000)
        # Store reasoning level, defaulting to None if not present
        self.reasoning_level = openai_config.get('reasoning_level') # Already loaded as None if missing

        if not self.base_url or not self.model:
            raise LLMClientError("Missing 'base_url' or 'model' in 'openai' configuration.")

        try:
            # Use AsyncOpenAI for asynchronous operations
            self.client = openai.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                # Add timeouts? connect=5.0, read=10.0
            )
        except Exception as e:
            # Log before raising
            logger.exception("Error initializing OpenAI AsyncClient")
            raise LLMClientError(f"Error initializing OpenAI AsyncClient: {e}")

    async def get_summary_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Calls the OpenAI-compatible API asynchronously with streaming and yields response chunks.

        Args:
            prompt: The full prompt string to send to the API.

        Yields:
            str: Chunks of the response content as they are received.

        Raises:
            LLMClientError: If an API error occurs during the request.
        """
        # Use logger.info or logger.debug
        logger.debug("Sending request to LLM API (stream)...")
        try:
            # Prepare base parameters
            params = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "stream": True,
            }
            # Conditionally add reasoning parameter
            if self.reasoning_level is not None:
                # Use logger.info or logger.debug
                logger.debug(f"Including reasoning level: {self.reasoning_level}")
                params['reasoning'] = self.reasoning_level

            # Make the API call with unpacked parameters
            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
            # Use logger.info or logger.debug
            logger.debug("End of LLM Stream.")
        except openai.APITimeoutError as e:
             logger.error(f"LLM API request timed out (stream): {e}")
             raise LLMClientError(f"LLM API request timed out: {e}")
        except openai.APIConnectionError as e:
            logger.error(f"Failed to connect to LLM API (stream): {e}")
            raise LLMClientError(f"Failed to connect to LLM API: {e}")
        except openai.RateLimitError as e:
            logger.error(f"LLM API request exceeded rate limit (stream): {e}")
            raise LLMClientError(f"LLM API request exceeded rate limit: {e}")
        except openai.APIStatusError as e:
             logger.error(f"LLM API returned an error status {e.status_code} (stream): {e}")
             raise LLMClientError(f"LLM API returned an error status {e.status_code}: {e}")
        except openai.APIError as e:
            logger.error(f"LLM API returned an API Error (stream): {e}")
            raise LLMClientError(f"LLM API returned an API Error: {e}")
        except Exception as e:
            # Log exception details using logger
            logger.exception("An unexpected error occurred during streaming LLM API call")
            raise LLMClientError(f"An unexpected error occurred during LLM API call: {type(e).__name__}: {e}")

    async def get_summary_no_stream(self, prompt: str) -> str:
        """
        Calls the OpenAI-compatible API asynchronously without streaming.

        Args:
            prompt: The full prompt string to send to the API.

        Returns:
            str: The complete response content.

        Raises:
            LLMClientError: If an API error occurs or the response is empty.
        """
        # Use logger.info or logger.debug
        logger.debug("Sending request to LLM API (no stream)...")
        try:
            # Prepare base parameters
            params = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            # Conditionally add reasoning parameter
            if self.reasoning_level is not None:
                 # Use logger.info or logger.debug
                 logger.debug(f"Including reasoning level: {self.reasoning_level}")
                 params['reasoning'] = self.reasoning_level

            # Make the API call with unpacked parameters
            response = await self.client.chat.completions.create(**params)

            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if content:
                     # Use logger.info or logger.debug
                     logger.debug("Received full response from LLM API (no stream).")
                     return content.strip()
                else:
                    logger.error("LLM API returned an empty message content (no stream).")
                    raise LLMClientError("LLM API returned an empty message content (no stream).")
            else:
                 logger.error("LLM API returned no valid choices or message (no stream).")
                 raise LLMClientError("LLM API returned no valid choices or message (no stream).")

        except openai.APITimeoutError as e:
             logger.error(f"LLM API request timed out (no stream): {e}")
             raise LLMClientError(f"LLM API request timed out (no stream): {e}")
        except openai.APIConnectionError as e:
            logger.error(f"Failed to connect to LLM API (no stream): {e}")
            raise LLMClientError(f"Failed to connect to LLM API (no stream): {e}")
        except openai.RateLimitError as e:
            logger.error(f"LLM API request exceeded rate limit (no stream): {e}")
            raise LLMClientError(f"LLM API request exceeded rate limit (no stream): {e}")
        except openai.APIStatusError as e:
             logger.error(f"LLM API returned an error status {e.status_code} (no stream): {e}")
             raise LLMClientError(f"LLM API returned an error status {e.status_code} (no stream): {e}")
        except openai.APIError as e:
            logger.error(f"LLM API returned an API Error (no stream): {e}")
            raise LLMClientError(f"LLM API returned an API Error (no stream): {e}")
        except Exception as e:
            # Log exception details using logger
            logger.exception("An unexpected error occurred during non-streaming LLM API call")
            raise LLMClientError(f"An unexpected error occurred during non-streaming LLM API call: {type(e).__name__}: {e}")

    async def close(self):
        """Closes the underlying HTTPX client if necessary."""
        # Use logger.info or logger.debug
        logger.debug("LLMClient close called (currently a no-op). Ensure resources are managed.")
        # Starting with openai v1.0+, explicit closing of the client might not be
        # strictly necessary as it uses httpx which manages connections.
        # However, if issues arise, investigate closing the internal httpx client.
        # For now, we assume the context management or garbage collection handles it.
        # Example if direct access was possible: await self.client._client.aclose()
        pass

# Example Usage Block (keep for potential testing)
async def _test_llm_client():
    # Mock config for testing - REPLACE WITH YOUR ACTUAL TEST SETUP
    mock_config = {
        "openai": {
            "base_url": "http://localhost:11434/v1", # Example: Ollama endpoint
            "model": "llama3",  # Example: A model served by Ollama
            "api_key": "NA", # Often not needed for local models
            "temperature": 0.1,
            "max_tokens": 100
        }
    }
    prompt = "Explain the concept of asynchronous programming in Python in one sentence."
    client = None # Initialize client to None
    try:
        client = LLMClient(mock_config)
        print(f"Initialized client for {client.base_url} with model {client.model}")

        full_response = ""
        async for chunk in client.get_summary_stream(prompt):
            print(chunk, end='', flush=True)
            full_response += chunk
        print("\n--- Full Response Received ---")
        print(full_response)

    except LLMClientError as e:
        print(f"\nClient Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
    finally:
        if client:
            await client.close()

if __name__ == '__main__':
    import asyncio
    # To run this test:
    # 1. Make sure you have an OpenAI-compatible server running (e.g., Ollama)
    #    at the 'base_url' specified in mock_config.
    # 2. Make sure the specified 'model' is available on that server.
    # 3. Uncomment the line below.
    # asyncio.run(_test_llm_client())
    print("LLMClient defined. Uncomment the asyncio.run line in __main__ to test.") 