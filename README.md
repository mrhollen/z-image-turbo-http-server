# Z-Image-Turbo HTTP Server

A lightweight HTTP server for the [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) text-to-image model, built with Python and `diffusers`.

## Features

- **Text-to-Image Generation**: Generate high-quality images from text prompts.
- **Web Interface**: Simple, user-friendly HTML interface for interacting with the model.
- **Custom Resolution**: Support for variable image dimensions (must be divisible by 16).
- **Seed Control**: Specify a seed for reproducible generations, or use random seeds.
- **Low Memory Mode**: Optional CPU offload for running on GPUs with limited VRAM.

## Prerequisites

- Python 3.8 or higher.
- CUDA-capable GPU (recommended) or CPU (slow).

## Installation

1.  Clone the repository (if applicable) or navigate to the project directory.
2.  Run the setup script to create a virtual environment and install dependencies:

    ```bash
    ./setup.sh
    ```

    This script will:
    - Create a `venv` directory.
    - Install `torch`, `diffusers`, `transformers`, and `accelerate`.

## Usage

### Starting the Server

1.  Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```
2.  Run the server:
    ```bash
    python server.py
    ```

### Command Line Arguments

- `--port`: Port to listen on (default: `8000`).
- `--device`: Device to use (default: `cuda`).
- `--low-mem`: Enable model CPU offload for lower VRAM usage.

Example:
```bash
python server.py --port 8080 --low-mem
```

### Using the Web UI

Once the server is running, open your web browser and navigate to:
`http://localhost:8000` (or the port you specified).

You can:
- Enter a text prompt.
- Adjust Width and Height.
- Set a Seed (use `-1` for random).
- Generate and download images.

## API Documentation

### `POST /generate`

Generates an image based on the provided parameters.

**Request Headers:**
- `Content-Type: application/json`

**Request Body (JSON):**
| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | string | **Required** | The text description of the image to generate. |
| `width` | integer | `1024` | Image width (must be divisible by 16). |
| `height` | integer | `1024` | Image height (must be divisible by 16). |
| `seed` | integer | `-1` | Random seed. Set to a specific number for reproducibility. |

**Response:**
- **Success (200 OK)**: Returns the generated image as a binary PNG stream.
- **Error (400 Bad Request)**: Invalid JSON, missing prompt, or invalid resolution.
- **Error (500 Internal Server Error)**: Server-side generation failure.

**Example `curl` Request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A futuristic city", "width": 512, "height": 512, "seed": 42}' \
  --output generated_image.png
```
