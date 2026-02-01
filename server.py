"""
OpenAI MCP Server for Claude Code
Provides DALL-E image generation and GPT chat completion tools.
"""

import os
import base64
from pathlib import Path
from mcp.server.fastmcp import FastMCP

try:
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError
except ImportError:
    raise ImportError("Please install openai: pip install openai")

# Initialize MCP server
mcp = FastMCP("openai-integration")

# Valid parameter options
DALLE3_SIZES = {"1024x1024", "1792x1024", "1024x1792"}
DALLE2_SIZES = {"256x256", "512x512", "1024x1024"}
QUALITIES = {"standard", "hd"}
STYLES = {"vivid", "natural"}
GPT_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}
VISION_MODELS = {"gpt-4o", "gpt-4o-mini"}


def get_client() -> OpenAI:
    """Get OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def handle_api_error(e: Exception) -> str:
    """Convert API errors to user-friendly messages."""
    if isinstance(e, RateLimitError):
        return "Error: Rate limit exceeded. Please wait and try again."
    elif isinstance(e, APIConnectionError):
        return "Error: Could not connect to OpenAI API. Check your internet connection."
    elif isinstance(e, APIError):
        return f"Error: OpenAI API error - {str(e)}"
    else:
        return f"Error: {str(e)}"


# =============================================================================
# DALL-E Image Generation Tools
# =============================================================================

@mcp.tool()
def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid"
) -> str:
    """
    Generate an image using OpenAI's DALL-E 3 API.

    Args:
        prompt: A detailed description of the image to generate
        size: Image dimensions - "1024x1024", "1792x1024", or "1024x1792"
        quality: "standard" or "hd" (hd costs more but has finer details)
        style: "vivid" (hyper-real/dramatic) or "natural" (more natural, less hyper-real)

    Returns:
        URL of the generated image
    """
    # Validate parameters
    if size not in DALLE3_SIZES:
        return f"Error: Invalid size '{size}'. Valid options: {', '.join(DALLE3_SIZES)}"
    if quality not in QUALITIES:
        return f"Error: Invalid quality '{quality}'. Valid options: {', '.join(QUALITIES)}"
    if style not in STYLES:
        return f"Error: Invalid style '{style}'. Valid options: {', '.join(STYLES)}"

    try:
        client = get_client()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
        )

        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt

        return f"Image URL: {image_url}\n\nRevised prompt used: {revised_prompt}"

    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
def generate_image_and_save(
    prompt: str,
    output_path: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid"
) -> str:
    """
    Generate an image using DALL-E 3 and save it to a local file.

    Args:
        prompt: A detailed description of the image to generate
        output_path: Local file path to save the image (e.g., "./output.png")
        size: Image dimensions - "1024x1024", "1792x1024", or "1024x1792"
        quality: "standard" or "hd"
        style: "vivid" or "natural"

    Returns:
        Path to the saved image file
    """
    import urllib.request

    # Validate parameters
    if size not in DALLE3_SIZES:
        return f"Error: Invalid size '{size}'. Valid options: {', '.join(DALLE3_SIZES)}"
    if quality not in QUALITIES:
        return f"Error: Invalid quality '{quality}'. Valid options: {', '.join(QUALITIES)}"
    if style not in STYLES:
        return f"Error: Invalid style '{style}'. Valid options: {', '.join(STYLES)}"

    try:
        client = get_client()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
        )

        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt

        # Download and save the image
        output_file = Path(output_path).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(image_url, str(output_file))

        return f"Image saved to: {output_file}\n\nRevised prompt used: {revised_prompt}"

    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
def edit_image(
    image_path: str,
    prompt: str,
    mask_path: str = None,
    size: str = "1024x1024"
) -> str:
    """
    Edit an existing image using DALL-E 2 (inpainting).

    Args:
        image_path: Path to the source image (must be PNG, square, less than 4MB)
        prompt: Description of the desired edit
        mask_path: Optional path to mask image (transparent areas will be edited)
        size: Output size - "256x256", "512x512", or "1024x1024"

    Returns:
        URL of the edited image
    """
    # Validate parameters
    if size not in DALLE2_SIZES:
        return f"Error: Invalid size '{size}'. Valid options: {', '.join(DALLE2_SIZES)}"

    image_file = Path(image_path).expanduser().resolve()
    if not image_file.exists():
        return f"Error: Image file not found: {image_file}"

    try:
        client = get_client()

        # Use context managers to properly handle file resources
        with open(image_file, "rb") as img:
            kwargs = {
                "model": "dall-e-2",
                "image": img,
                "prompt": prompt,
                "size": size,
                "n": 1,
            }

            if mask_path:
                mask_file = Path(mask_path).expanduser().resolve()
                if not mask_file.exists():
                    return f"Error: Mask file not found: {mask_file}"
                with open(mask_file, "rb") as mask:
                    kwargs["mask"] = mask
                    response = client.images.edit(**kwargs)
            else:
                response = client.images.edit(**kwargs)

        return f"Edited image URL: {response.data[0].url}"

    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
def create_image_variation(
    image_path: str,
    n: int = 1,
    size: str = "1024x1024"
) -> str:
    """
    Create variations of an existing image using DALL-E 2.

    Args:
        image_path: Path to the source image (must be PNG, square, less than 4MB)
        n: Number of variations to generate (1-10)
        size: Output size - "256x256", "512x512", or "1024x1024"

    Returns:
        URLs of the generated variations
    """
    # Validate parameters
    if size not in DALLE2_SIZES:
        return f"Error: Invalid size '{size}'. Valid options: {', '.join(DALLE2_SIZES)}"
    if not 1 <= n <= 10:
        return "Error: n must be between 1 and 10"

    image_file = Path(image_path).expanduser().resolve()
    if not image_file.exists():
        return f"Error: Image file not found: {image_file}"

    try:
        client = get_client()

        # Use context manager to properly handle file resources
        with open(image_file, "rb") as img:
            response = client.images.create_variation(
                model="dall-e-2",
                image=img,
                n=n,
                size=size,
            )

        urls = [f"Variation {i+1}: {img.url}" for i, img in enumerate(response.data)]
        return "\n".join(urls)

    except Exception as e:
        return handle_api_error(e)


# =============================================================================
# GPT Chat Completion Tools
# =============================================================================

@mcp.tool()
def chat_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-4o",
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> str:
    """
    Generate a chat completion using OpenAI's GPT models.

    Args:
        prompt: The user message/question to send
        system_message: System instructions for the model's behavior
        model: Model to use - "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
        max_tokens: Maximum tokens in the response (default 2048)
        temperature: Creativity level 0.0-2.0 (lower = more focused, higher = more creative)

    Returns:
        The model's response text
    """
    # Validate parameters
    if model not in GPT_MODELS:
        return f"Error: Invalid model '{model}'. Valid options: {', '.join(GPT_MODELS)}"
    if not 0.0 <= temperature <= 2.0:
        return "Error: temperature must be between 0.0 and 2.0"
    if max_tokens < 1:
        return "Error: max_tokens must be at least 1"

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
def analyze_image_with_gpt(
    image_path: str,
    prompt: str = "Describe this image in detail.",
    model: str = "gpt-4o"
) -> str:
    """
    Analyze an image using GPT-4 Vision capabilities.

    Args:
        image_path: Path to the image file to analyze
        prompt: Question or instruction about the image
        model: Model to use - "gpt-4o" or "gpt-4o-mini" (must support vision)

    Returns:
        The model's analysis of the image
    """
    # Validate parameters
    if model not in VISION_MODELS:
        return f"Error: Invalid model '{model}'. Valid vision models: {', '.join(VISION_MODELS)}"

    image_file = Path(image_path).expanduser().resolve()
    if not image_file.exists():
        return f"Error: Image file not found: {image_file}"

    # Determine media type
    suffix = image_file.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    media_type = media_types.get(suffix)
    if not media_type:
        return f"Error: Unsupported image format '{suffix}'. Supported: {', '.join(media_types.keys())}"

    try:
        client = get_client()

        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2048,
        )

        return response.choices[0].message.content

    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
def summarize_text(
    text: str,
    style: str = "concise",
    model: str = "gpt-4o-mini"
) -> str:
    """
    Summarize text using GPT.

    Args:
        text: The text to summarize
        style: Summary style - "concise" (brief), "detailed" (comprehensive), or "bullets" (bullet points)
        model: Model to use (gpt-4o-mini is cost-effective for summarization)

    Returns:
        The summarized text
    """
    # Validate parameters
    valid_styles = {"concise", "detailed", "bullets"}
    if style not in valid_styles:
        return f"Error: Invalid style '{style}'. Valid options: {', '.join(valid_styles)}"
    if model not in GPT_MODELS:
        return f"Error: Invalid model '{model}'. Valid options: {', '.join(GPT_MODELS)}"

    style_instructions = {
        "concise": "Provide a brief, concise summary in 2-3 sentences.",
        "detailed": "Provide a comprehensive summary covering all main points.",
        "bullets": "Summarize the key points as a bullet-point list."
    }

    try:
        client = get_client()
        system_msg = f"You are a summarization assistant. {style_instructions[style]}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
            ],
            max_tokens=1024,
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
def translate_text(
    text: str,
    target_language: str,
    source_language: str = "auto",
    model: str = "gpt-4o-mini"
) -> str:
    """
    Translate text to a target language using GPT.

    Args:
        text: The text to translate
        target_language: The language to translate to (e.g., "Spanish", "French", "Japanese")
        source_language: Source language or "auto" for auto-detection
        model: Model to use

    Returns:
        The translated text
    """
    # Validate parameters
    if model not in GPT_MODELS:
        return f"Error: Invalid model '{model}'. Valid options: {', '.join(GPT_MODELS)}"
    if not target_language.strip():
        return "Error: target_language cannot be empty"

    try:
        client = get_client()

        if source_language == "auto":
            instruction = f"Translate the following text to {target_language}. Preserve the original tone and meaning."
        else:
            instruction = f"Translate the following text from {source_language} to {target_language}. Preserve the original tone and meaning."

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional translator. Provide only the translation without explanations."},
                {"role": "user", "content": f"{instruction}\n\nText:\n{text}"}
            ],
            max_tokens=2048,
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return handle_api_error(e)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()
