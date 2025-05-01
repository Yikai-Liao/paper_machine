import json
import sys
import re # Import regex module
from loguru import logger # Ensure loguru logger is used

class SummaryParseError(Exception):
    """Custom exception for summary parsing errors."""
    pass

def extract_summary_json(response: str, separator: str = '----') -> dict:
    """
    Extracts the JSON part from the API response robustly and parses it,
    ensuring the result is a dictionary.
    Tries finding the separator first, then falls back to finding the first
    valid JSON object within the whole response if the separator is missing
    or parsing fails after the separator.

    Args:
        response: The full string response from the LLM API.
        separator: The string that ideally separates introductory text from the JSON payload.

    Returns:
        A dictionary parsed from the JSON string.

    Raises:
        SummaryParseError: If no valid JSON dictionary can be extracted.
    """
    parsed_json = None
    error_messages = []

    # Attempt 1: Try using the separator
    if separator in response:
        parts = response.split(separator, 1)
        if len(parts) == 2:
            json_text_potential = parts[1].strip()
            try:
                # Try parsing the text after separator directly
                parsed_json = json.loads(json_text_potential)
                if isinstance(parsed_json, dict):
                    return parsed_json
                else:
                    error_messages.append(f"Parsed JSON after separator is not a dictionary (type: {type(parsed_json)}).")
                    parsed_json = None # Reset for fallback
            except json.JSONDecodeError as e_sep:
                # If direct parsing fails, try finding {..} within the part after separator
                match = re.search(r'\{.*\}', json_text_potential, re.DOTALL)
                if match:
                    json_braces_text = match.group(0)
                    try:
                        parsed_json = json.loads(json_braces_text)
                        if isinstance(parsed_json, dict):
                            return parsed_json
                        else:
                            error_messages.append(f"Parsed JSON (in braces) after separator is not a dictionary (type: {type(parsed_json)}).")
                            parsed_json = None # Reset for fallback
                    except json.JSONDecodeError as e_brace_sep:
                         error_messages.append(f"Could not parse JSON (in braces) after separator: {e_brace_sep}")
                else:
                    error_messages.append(f"Could not parse JSON after separator (direct parse failed: {e_sep}) and no {{...}} block found after separator.")
            except Exception as e_gen_sep: # Catch other errors after separator
                 error_messages.append(f"Unexpected error parsing after separator: {e_gen_sep}")
        else:
             # This case should not happen if separator is in response, but handle defensively
             error_messages.append("Separator found, but split did not yield two parts.")
    else:
        error_messages.append(f"Separator '{separator}' not found in the response.")

    # Attempt 2: Fallback - Search for the first { ... } block in the *entire* response
    # Use regex to find the first potential JSON object (non-greedy match might be safer)
    # This regex tries to find the first '{' and the last '}' that form a potentially valid structure.
    # It's not perfect for nested structures but often works for LLM outputs.
    # A more robust approach might involve iterating and checking brace balance.
    match = re.search(r'^.*?(\{.*\}).*$', response, re.DOTALL)
    if match:
        json_text_full_response = match.group(1).strip() # Extract the matched JSON part
        try:
            parsed_json = json.loads(json_text_full_response)
            if isinstance(parsed_json, dict):
                logger.warning("Extracted JSON by finding {..} in the full response (separator method failed).")
                return parsed_json
            else:
                 error_messages.append(f"Parsed JSON found in full response is not a dictionary (type: {type(parsed_json)}).")
        except json.JSONDecodeError as e_full:
            error_messages.append(f"Could not parse the {{...}} block found in the full response: {e_full}")
        except Exception as e_gen_full:
             error_messages.append(f"Unexpected error parsing {{...}} block from full response: {e_gen_full}")
    else:
        error_messages.append("Could not find any {{...}} block in the full response.")

    # If all attempts failed
    raise SummaryParseError("Failed to extract a valid JSON dictionary from the response. Errors encountered: " + "; ".join(error_messages))


# Example Usage (for testing)
if __name__ == '__main__':
    test_response_ok = """
Some introductory text from the LLM.
It might have multiple lines.
----
{
    "title": "Test Paper Summary",
    "key_points": [
        "Point 1",
        "Point 2"
    ],
    "confidence": 0.9
}
Some trailing text maybe.
    """
    test_response_bad_json = """
Analysis here.
----
{
    "title": "Bad JSON",
    "key_points": [ "Point 1", "Point 2"
    "missing_comma": true
}
    """
    test_response_no_separator_ok_json = """
Some text first.
{
    "title": "No Separator",
    "key_points": ["A"],
    "valid": true
}
More text after.
    """
    test_response_no_separator_bad_json = """
{
    "title": "No Sep Bad JSON",
    "error":
}
    """
    test_response_no_separator_no_json = "Just plain text, no JSON here."
    test_response_no_dict = """
Separator ----
["just", "a", "list"]
    """
    test_response_tricky_braces = """
Blah ---- { "outer": { "inner": "value" } } { "ignored": "stuff" }
    """
    test_response_only_json = '{"key": "value only"}'

    tests = {
        "OK_With_Separator": test_response_ok,
        "Bad_JSON_With_Separator": test_response_bad_json,
        "OK_JSON_No_Separator": test_response_no_separator_ok_json,
        "Bad_JSON_No_Separator": test_response_no_separator_bad_json,
        "No_JSON_No_Separator": test_response_no_separator_no_json,
        "Not_Dict_With_Separator": test_response_no_dict,
        "Tricky_Braces_With_Separator": test_response_tricky_braces,
        "Only_JSON_No_Separator": test_response_only_json
    }

    for name, response in tests.items():
        print(f"\n--- Testing: {name} ---")
        try:
            summary = extract_summary_json(response)
            print("Parsed Summary:", json.dumps(summary, indent=2))
        except SummaryParseError as e:
            print(f"Error: {e}", file=sys.stderr) 