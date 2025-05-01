import pymupdf4llm # Import the new library
from pathlib import Path
import sys
# Replace standard logging with loguru
# import logging
import traceback # For detailed error logging
from loguru import logger # Import loguru
import io # For capturing stderr
import contextlib # For redirect_stderr
import re # For matching MuPDF errors

# Import the save_file utility
from .utils import save_file

# Configure loguru (optional, can also be configured in main entry point)
# Example configuration (adjust as needed):
# logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
# logger.add("file_{time}.log", level="DEBUG") # Log to file

# Keep FitzError import from base PyMuPDF for specific error handling if needed,
# although pymupdf4llm might handle some errors internally.
import fitz
try:
    FitzError = fitz.fitz.FitzError
except AttributeError:
    try:
        FitzError = fitz.FitzError
    except AttributeError:
        logger.warning("Could not find FitzError in fitz module. PDF processing might be unreliable.")
        class FitzError(Exception): pass

# Precompile the regex for efficiency
MUPDF_COLOR_SPACE_ERROR_RE = re.compile(r"MuPDF error: syntax error: could not parse color space", re.IGNORECASE)

def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """
    Extracts text content from a given PDF file as Markdown using pymupdf4llm,
    capturing and logging specific MuPDF errors from stderr as warnings.

    Args:
        pdf_path: The Path object pointing to the PDF file.

    Returns:
        A string containing the extracted text in Markdown format,
        or None if a critical error occurs or the file cannot be processed.
    """
    pdf_str = str(pdf_path)
    logger.info(f"Attempting Markdown extraction for: {pdf_str}")

    # Capture stderr during the pymupdf4llm call
    stderr_capture = io.StringIO()
    md_text = None
    extraction_error = None

    try:
        with contextlib.redirect_stderr(stderr_capture):
            # Use pymupdf4llm to directly get Markdown text
            md_text = pymupdf4llm.to_markdown(pdf_str)

    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_str}")
        return None
    except FitzError as fe:
        logger.error(f"PyMuPDF FitzError during Markdown extraction for {pdf_str}: {fe}")
        # Log traceback for more details on PyMuPDF errors
        # logger.opt(exception=True).error("FitzError details:")
        extraction_error = fe
        # Return None below after checking stderr
    except Exception as e:
        logger.error(f"An unexpected error occurred during Markdown extraction call for {pdf_str}: {e}")
        logger.opt(exception=True).error("Unexpected extraction error details:")
        extraction_error = e
        # Return None below after checking stderr

    # Process captured stderr output
    stderr_output = stderr_capture.getvalue()
    if stderr_output:
        lines = stderr_output.strip().split('\n')
        for line in lines:
            if MUPDF_COLOR_SPACE_ERROR_RE.search(line):
                logger.warning(f"[MuPDF Color Space] {line.strip()} (PDF: {pdf_str})")
            else:
                # Log other stderr output as error or debug
                logger.error(f"[stderr PDF Extract] {line.strip()} (PDF: {pdf_str})")

    # Handle results after checking stderr
    if extraction_error:
         logger.warning(f"Extraction failed for {pdf_str} due to error: {extraction_error}. Returning None.")
         return None # Extraction itself failed critically

    if md_text is None:
        # This might happen if pymupdf4llm returns None without raising an exception
        logger.warning(f"Markdown extraction returned None unexpectedly for: {pdf_str}. Returning None.")
        return None

    # Check if the output seems valid (not empty)
    if not md_text.strip():
        logger.warning(f"Markdown extraction resulted in empty content for: {pdf_str}")
        return "" # Return empty string (consistent with successful but empty doc)

    logger.info(f"Successfully extracted Markdown from: {pdf_str}")
    return md_text

# --- Process Pool Worker Function ---
def extract_and_save_markdown_worker(pdf_path: Path, output_dir: Path, save_paper_md_flag: bool) -> Path | None:
    """
    Worker function for the process pool. Extracts text from a PDF and optionally saves it.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save the markdown file.
        save_paper_md_flag: Boolean indicating whether to save the extracted markdown.

    Returns:
        The original pdf_path if extraction was successful (even if saving wasn't needed or failed),
        otherwise None.
    """
    # Note: Loguru might need reconfiguration in each new process if not using
    # specific inter-process logging handlers. Basic file/stderr logging might work,
    # but be mindful of potential conflicts or mixed output.
    # Consider passing basic info back for the main process to log.

    pdf_name = pdf_path.name
    logger.info(f"[Worker:{pdf_name}] Starting extraction...") # Simple worker log

    extracted_text = extract_text_from_pdf(pdf_path)

    if extracted_text is None:
        logger.error(f"[Worker:{pdf_name}] Text extraction failed critically.")
        return None # Indicate critical failure
    elif not extracted_text:
         logger.warning(f"[Worker:{pdf_name}] Text extraction resulted in empty content.")
         # Return path even if empty, as extraction technically didn't error
         return pdf_path

    logger.info(f"[Worker:{pdf_name}] Text extraction successful.")

    output_filename_paper = output_dir / f"{pdf_path.stem}_paper.md"
    save_needed = True # Assume we need to save by default

    if save_paper_md_flag:
        logger.debug(f"[Worker:{pdf_name}] Saving extracted text (markdown_paper=true) to {output_filename_paper}")
    else:
        # Even if markdown_paper is false, we need the temp file for the LLM stage
        logger.debug(f"[Worker:{pdf_name}] Saving temporary extracted text (markdown_paper=false) to {output_filename_paper}")

    # Always save the file (either permanently or temporarily)
    if not save_file(output_filename_paper, extracted_text, "Extracted Paper Markdown"):
        logger.error(f"[Worker:{pdf_name}] Failed to save extracted paper markdown to {output_filename_paper}.")
        # If saving fails, we cannot proceed with this PDF for the LLM stage
        return None # Indicate critical failure (cannot proceed without the file)
    else:
        logger.info(f"[Worker:{pdf_name}] Successfully saved extracted paper markdown to {output_filename_paper}.")

    return pdf_path # Return path on successful extraction and save
# ------------------------------------

# Note: The previous complex Fitz object handling (is_pdf, is_encrypted)
# is now implicitly handled within pymupdf4llm.to_markdown().
# Error handling focuses on the outcome of that function call.