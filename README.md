# OpenAI MCP Server for Claude Code

An MCP (Model Context Protocol) server that integrates OpenAI APIs with Claude Code, providing DALL-E image generation and GPT chat completion capabilities.

## Features

### Image Generation (DALL-E)
- `generate_image` - Generate images from text prompts using DALL-E 3
- `generate_image_and_save` - Generate and save images locally
- `edit_image` - Edit/inpaint existing images using DALL-E 2
- `create_image_variation` - Create variations of an existing image

### Chat/Text (GPT)
- `chat_completion` - Chat completions with GPT-4o, GPT-4o-mini, etc.
- `analyze_image_with_gpt` - Analyze images using GPT-4 Vision
- `summarize_text` - Summarize text (concise, detailed, or bullet points)
- `translate_text` - Translate text between languages

## Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Claude Code CLI

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fryzhykau/openai-mcp-server.git
cd openai-mcp-server
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add to Claude Code

```bash
claude mcp add openai-tools -e OPENAI_API_KEY=sk-your-key-here -- python /path/to/openai-mcp-server/server.py
```

Replace `/path/to/openai-mcp-server/server.py` with the actual path to the server.

### 4. Verify installation

In Claude Code, run `/mcp` to see the server and available tools.

## Usage Examples

Once configured, use naturally in Claude Code:

```
> Generate an image of a cyberpunk cityscape at night
> Save an image of a mountain landscape to ./output.png
> Analyze this screenshot and tell me what's wrong
> Translate this text to Spanish
> Summarize this article in bullet points
> Ask GPT-4 to explain quantum computing
```

## Configuration Scopes

When adding the MCP server, you can choose different scopes:

| Scope | Flag | Description |
|-------|------|-------------|
| Local | (default) | Only you, only this project |
| Project | `--scope project` | Shared with team via `.mcp.json` |
| User | `--scope user` | Available in all your projects |

## API Parameters

### DALL-E 3 (generate_image, generate_image_and_save)
- **size**: `1024x1024`, `1792x1024`, `1024x1792`
- **quality**: `standard`, `hd`
- **style**: `vivid`, `natural`

### DALL-E 2 (edit_image, create_image_variation)
- **size**: `256x256`, `512x512`, `1024x1024`

### GPT Models
- `gpt-4o` - Most capable, recommended for complex tasks
- `gpt-4o-mini` - Cost-effective, good for simple tasks
- `gpt-4-turbo` - Previous generation, still capable
- `gpt-3.5-turbo` - Fastest, most economical

## Security Notes

- API keys are passed via environment variables, never hardcoded
- All user inputs are validated before API calls
- File operations use resolved absolute paths
- API errors are caught and returned as user-friendly messages

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
