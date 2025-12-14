# Z-Image-Turbo HTTP Server

A lightweight HTTP server for the [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) text-to-image model, built with Python and `diffusers`.

## Features

- **Text-to-Image Generation**: Generate high-quality images from text prompts.
- **Web Interface**: Simple HTML UI with prompt, width/height, aspect-ratio helper, and seed inputs.
- **Custom Resolution**: Variable dimensions (must be divisible by 16). Helper warnings suggest the nearest valid size.
- **Seed Control**: Specify a seed for reproducible generations, or use random seeds (`-1`).
- **Low Memory Mode**: Optional CPU offload for running on GPUs with limited VRAM.
- **Gallery**: Optional on-disk saving with an infinite-scroll gallery when `--save-dir` is provided.

## Prerequisites

- Python 3.8 or higher.
- CUDA-capable GPU (recommended); CPU works but is slow.

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
    python server.py [--port 8000] [--device cuda] [--low-mem] [--save-dir ./outputs]
    ```

### Command Line Arguments

- `--port`: Port to listen on (default: `8000`).
- `--device`: Device to use (default: `cuda`).
- `--low-mem`: Enable model CPU offload for lower VRAM usage.
- `--save-dir`: Directory to write generated images (enables the gallery UI/endpoints).

Example:
```bash
python server.py --port 8080 --low-mem --save-dir ./generated
```

### Using the Web UI

Once the server is running, open your web browser and navigate to:
`http://localhost:8000` (or the port you specified).

You can:
- Enter a text prompt.
- Adjust Width and Height (must be divisible by 16; helper suggests the nearest valid size).
- Optionally enter an aspect ratio (e.g., `16:9`) to auto-calculate the other dimension.
- Set a Seed (use `-1` for random).
- Generate and download images.
- Scroll through saved images in the gallery when `--save-dir` is set (infinite scroll).

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
- **Success (200 OK)**: Returns the generated image as a binary PNG stream. Images are also written to `--save-dir` if provided (filename format: `generated_<timestamp>_<seed>.png`).
- **Error (400 Bad Request)**: Invalid JSON, missing prompt, or invalid resolution.
- **Error (500 Internal Server Error)**: Server-side generation failure.

**Generation settings:** `num_inference_steps=8`, `guidance_scale=0.0`, `torch_dtype=torch.bfloat16`. Seeded runs use a `torch.Generator` on the configured device.

**Example `curl` Request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A futuristic city", "width": 512, "height": 512, "seed": 42}' \
  --output generated_image.png
```

### `GET /gallery/images` (requires `--save-dir`)
List saved images with pagination.

**Query params:**
- `limit` (default `20`): Max results to return.
- `offset` (default `0`): Starting index.

**Response (200 OK):**
```json
{ "images": ["generated_...png"], "total": 42, "has_more": true }
```

### `GET /gallery/image/<filename>` (requires `--save-dir`)
Returns a saved image by filename (PNG). Basic filename safety checks are applied.
