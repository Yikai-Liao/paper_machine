from abc import ABC, abstractmethod
from typing import List
import numpy as np
from loguru import logger
import tempfile
import json
import os
import time
import requests
from openai import OpenAI

class EmbedBase(ABC):
    @abstractmethod
    def embed(self, text: List[str]) -> np.ndarray[np.float32, np.float32]:
        raise NotImplementedError("Subclasses must implement this method")
    
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError("Subclasses must implement this method")
    
# class JasperLocalEmbed(EmbedBase):
#     def __init__(self) -> None:
#         super().__init__()
    

class AliCloudEmbed(EmbedBase):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v3",
        dim: int = 1024,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._dim = dim
        
        # lazy import here to avoid unnecessary dependencies
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def dim(self) -> int:
        return self._dim

    def _prepare_batch_input(self, text: List[str]) -> List[dict]:
        """Prepares the batch input data structure."""
        return [
            {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": self.model,
                    "input": data,
                    "encoding_format": "float",
                    "dimensions": self.dim,
                },
            }
            for i, data in enumerate(text)
        ]

    def _upload_batch_file(self, batch_input: List[dict]) -> str:
        """Writes batch input to a temporary file and uploads it, returning the file ID."""
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as tmpf:
                tmp_path = tmpf.name
                for item in batch_input:
                    tmpf.write(json.dumps(item, ensure_ascii=False) + "\n")
                tmpf.flush()
            
            # Log the content before uploading
            try:
                with open(tmp_path, "r", encoding="utf-8") as read_tmpf:
                    file_content = read_tmpf.read()
                    logger.debug(f"Content of temporary batch file ({tmp_path}):\n{file_content}")
            except Exception as e:
                 logger.warning(f"Could not read temporary file {tmp_path} for logging: {e}")

            # Upload the file
            with open(tmp_path, "rb") as f:
                file_obj = self.client.files.create(file=f, purpose="batch")
            
            return file_obj.id
        finally:
            # Clean up the temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {tmp_path}: {e}")

    def _create_and_poll_batch_job(self, input_file_id: str) -> tuple[str, str | None]:
        """Creates a batch job and polls until completion. Returns batch status and output file ID."""
        response = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )

        batch_id = response.id
        max_wait = 600  # seconds
        interval = 5  # seconds
        waited = 0

        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            status = getattr(batch_status, "status", None)
            logger.debug(f"Batch {batch_id} status: {status}")

            if status == "completed":
                output_file_id = getattr(batch_status, "output_file_id", None)
                if not output_file_id:
                   logger.error(f"Batch {batch_id} completed but no output file ID found.")
                   raise RuntimeError(f"Batch {batch_id} completed but no output file ID.")
                return status, output_file_id
            elif status in ("failed", "cancelled", "expired"):
                logger.error(f"Batch {batch_id} failed or was cancelled/expired with status: {status}")
                raise RuntimeError(f"Batch job {batch_id} ended with status: {status}")
            
            time.sleep(interval)
            waited += interval
            if waited > max_wait:
                logger.error(f"Batch {batch_id} timed out after {max_wait} seconds.")
                raise TimeoutError(f"Batch {batch_id} did not complete in {max_wait} seconds.")

    def _download_and_decode_output(self, output_file_id: str) -> str:
        """Downloads (using content API) and decodes the batch job output file."""
        try:
            logger.debug(f"Downloading content for file ID: {output_file_id}")
            # Directly use the content API endpoint
            file_content_response = self.client.files.content(output_file_id)

            # Handle potential streaming response or direct bytes
            if hasattr(file_content_response, "read"):
                # If it's a file-like object (e.g., httpx response stream)
                file_bytes = file_content_response.read()
            elif isinstance(file_content_response, bytes):
                 # If it's already bytes
                 file_bytes = file_content_response
            else:
                 # Unexpected type, log an error
                 logger.error(f"Unexpected type for file content: {type(file_content_response)}")
                 raise TypeError(f"Unexpected type received for file content: {type(file_content_response)}")

            # Decode the bytes to text
            file_text = file_bytes.decode("utf-8")
            logger.debug(f"Successfully downloaded and decoded {len(file_bytes)} bytes for file {output_file_id}.")
            return file_text

        except Exception as e:
            logger.error(f"Failed to download or decode output file {output_file_id} using content API: {e}")
            # Re-raise the exception after logging
            raise

    def _parse_output_and_extract_embeddings(self, file_text: str) -> List[List[float]]:
        """Parses the decoded output text and extracts embeddings."""
        embeddings = []
        for line in file_text.strip().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line as JSON: {line} ({e})")
                continue

            emb = None
            try:
                # Primary extraction path
                resp_body = obj.get("response", {}).get("body", {})
                if resp_body: # Check if response.body exists
                    emb_data = resp_body.get("data", [])
                    if emb_data and isinstance(emb_data, list) and len(emb_data) > 0:
                         # Check if the first item has the embedding
                         first_item = emb_data[0]
                         if isinstance(first_item, dict):
                             emb = first_item.get("embedding")
            except Exception as e:
                logger.warning(f"Error accessing nested embedding in line: {line} ({e})")

            # Fallback attempts
            if emb is None:
                emb = obj.get("embedding")
            if emb is None and isinstance(obj.get("data"), dict): # Check if data is a dict before accessing get
                emb = obj.get("data", {}).get("embedding")
            # Additional fallback if 'data' itself is the list of embeddings (less likely based on logs but safe)
            if emb is None and isinstance(obj.get("data"), list) and len(obj.get("data")) > 0:
                 first_data_item = obj.get("data")[0]
                 if isinstance(first_data_item, dict):
                     emb = first_data_item.get("embedding")


            if emb is not None and isinstance(emb, list):
                embeddings.append(emb)
            else:
                logger.warning(f"No valid embedding found or extracted in line: {line}")
                # Optionally add more debug info about obj structure here if needed
                # logger.debug(f"Object structure: {obj}")

        return embeddings

    def embed(self, text: List[str]) -> np.ndarray[np.float32, np.float32]:
        """Generates embeddings for a list of texts using AliCloud batch API."""
        if not text:
            return np.empty((0, self.dim), dtype=np.float32)
            
        # 1. Prepare batch input
        batch_input = self._prepare_batch_input(text)

        # 2. Upload batch file
        input_file_id = self._upload_batch_file(batch_input)
        logger.debug(f"Uploaded batch input file with ID: {input_file_id}")

        # 3. Create and poll batch job
        status, output_file_id = self._create_and_poll_batch_job(input_file_id)
        logger.debug(f"Batch job completed with status: {status}, output file ID: {output_file_id}")
        
        if not output_file_id:
             # This case should be handled by _create_and_poll_batch_job raising an error,
             # but added as a safeguard.
             raise RuntimeError("Batch job finished but did not return an output file ID.")

        # 4. Download and decode output
        file_text = self._download_and_decode_output(output_file_id)

        # 5. Parse output and extract embeddings
        embeddings_list = self._parse_output_and_extract_embeddings(file_text)

        # Convert to 2D numpy array of float32
        if not embeddings_list:
             logger.warning("No embeddings were extracted from the batch output.")
             # Return empty array with correct dimensions if possible
             return np.empty((0, self.dim), dtype=np.float32)

        try:
            arr = np.array(embeddings_list, dtype=np.float32)
            # Verification step (optional but good practice)
            if arr.ndim != 2 or arr.shape[1] != self.dim:
                logger.error(f"Final embedding array shape mismatch: expected (N, {self.dim}), got {arr.shape}")
                # Decide how to handle mismatch: raise error or return empty/partial?
                # Raising error is safer to signal a problem.
                raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {arr.shape[1]}")
            logger.debug(f"Successfully created embedding array with shape: {arr.shape}")
            return arr
        except ValueError as e:
             # Catch potential errors during np.array creation (e.g., inconsistent list lengths)
             logger.error(f"Error converting extracted embeddings to NumPy array: {e}")
             logger.error(f"Problematic list: {embeddings_list[:5]}...") # Log snippet
             raise ValueError("Failed to create final embedding array due to inconsistent data.") from e

