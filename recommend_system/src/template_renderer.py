import sys
import re
import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound, TemplateSyntaxError

class TemplateError(Exception):
    """Custom exception for template loading or rendering errors."""
    pass

def _clean_filename_for_title(filename: str) -> str:
    """Cleans the filename (without extension) to be used as a title."""
    # Remove common summary suffixes
    title = re.sub(r'_summary$', '', filename, flags=re.IGNORECASE)
    # Replace underscores/hyphens with spaces
    title = title.replace('_', ' ').replace('-', ' ')
    # Simple title case (capitalize first letter of each word)
    # Consider using a more robust title casing library if needed
    title = title.title()
    # Remove leading/trailing whitespace
    return title.strip()

def _get_formatted_timestamp() -> str:
    """Generates a formatted timestamp string with timezone offset."""
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    # Format date: %Y-%m-%dT%H:%M:%S%z (ensure timezone offset is included)
    # Python's %z might not include colon, manually format offset for ISO 8601
    utc_offset = now.utcoffset()
    if utc_offset is not None:
        utc_offset_seconds = int(utc_offset.total_seconds())
        offset_sign = '+' if utc_offset_seconds >= 0 else '-'
        offset_hours = abs(utc_offset_seconds) // 3600
        offset_minutes_part = (abs(utc_offset_seconds) % 3600) // 60
        # ISO 8601 format for timezone offset (e.g., +05:30, -08:00, Z for UTC)
        if offset_hours == 0 and offset_minutes_part == 0:
            tz_str = "Z"
        else:
             tz_str = f"{offset_sign}{offset_hours:02d}:{offset_minutes_part:02d}"
    else:
        # Fallback if no timezone info (shouldn't happen with astimezone())
        tz_str = "Z" # Assume UTC if offset is None

    # Format without %z, then append manual offset
    return now.strftime('%Y-%m-%dT%H:%M:%S') + tz_str


class TemplateRenderer:
    """Loads a Jinja2 template and renders summaries."""

    def __init__(self, template_path_str: str):
        """
        Initializes the renderer by loading the Jinja2 template.

        Args:
            template_path_str: The path to the Jinja2 template file.

        Raises:
            TemplateError: If the template file is not found or has syntax errors.
        """
        template_path = Path(template_path_str)
        if not template_path.is_file():
            raise TemplateError(f"Jinja2 template file not found at {template_path}")

        try:
            template_dir = template_path.parent
            template_filename = template_path.name
            env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml', 'md']), # Enable autoescape for md
                trim_blocks=True, # Often useful for templates
                lstrip_blocks=True # Often useful for templates
            )
            self.template = env.get_template(template_filename)
        except TemplateNotFound:
            raise TemplateError(f"Jinja2 template '{template_filename}' not found in directory '{template_dir}'")
        except TemplateSyntaxError as e:
            raise TemplateError(f"Syntax error in Jinja2 template {template_path}: {e}")
        except Exception as e:
            raise TemplateError(f"Error loading Jinja2 template {template_path}: {e}")

    def render_summary(self, summary_data: dict, original_pdf_path: Path) -> str:
        """
        Renders the summary data using the loaded Jinja2 template.

        Args:
            summary_data: The dictionary containing the parsed summary information from the LLM.
            original_pdf_path: The Path object of the original PDF file (used for title generation).

        Returns:
            The rendered output as a string.

        Raises:
            TemplateError: If an error occurs during template rendering.
        """
        try:
            # Prepare the context for the template
            template_context = summary_data.copy() # Start with parsed JSON data

            # Add generated timestamp
            template_context['time'] = _get_formatted_timestamp()

            # Add original filename
            template_context['original_filename'] = original_pdf_path.name

            # Render the template
            rendered_output = self.template.render(template_context)
            return rendered_output

        except Exception as e:
            # Catch potential errors during render (e.g., missing variables in template)
            raise TemplateError(f"Error rendering template for {original_pdf_path.name}: {e}")

# Example Usage (for testing)
if __name__ == '__main__':
    # Create dummy files for testing
    test_dir = Path("_test_template_renderer")
    test_dir.mkdir(exist_ok=True)
    dummy_template_path = test_dir / "dummy_template.md.jinja"
    # Use a Path object for the dummy PDF path
    dummy_pdf_path = Path("some_paper_v1_final_summary.pdf")

    dummy_template_content = """
# {{ paper_title }}

**Original File:** `{{ original_filename }}`
**Processed Time:** {{ time }}

## Summary Points
{% if key_points %}
{% for point in key_points %}
- {{ point }}
{% endfor %}
{% else %}
No key points provided.
{% endif %}

**Confidence:** {{ confidence | default('N/A') }}
"""
    dummy_template_path.write_text(dummy_template_content, encoding='utf-8')

    mock_summary_data = {
        "key_points": ["Introduces method X.", "Achieves SOTA on benchmark Y.", "Future work includes Z."],
        "confidence": 0.85
        # Assume 'title' might be in the JSON but we override with filename based one
    }

    print(f"--- Testing TemplateRenderer with {dummy_template_path} ---")
    renderer = None
    try:
        renderer = TemplateRenderer(str(dummy_template_path))
        rendered = renderer.render_summary(mock_summary_data, dummy_pdf_path)
        print("\n--- Rendered Output ---")
        print(rendered)
        print("--- Test Successful ---")
    except TemplateError as e:
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error during test: {e}", file=sys.stderr)
    finally:
        # Clean up dummy file and dir
        # import shutil
        # if test_dir.exists():
        #      shutil.rmtree(test_dir)
         print(f"Cleanup: Remember to remove the {test_dir} directory and its contents ({dummy_template_path}).") 