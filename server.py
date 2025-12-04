import http.server
import json
import argparse
import io
import torch
import os
import time
import glob
from diffusers import DiffusionPipeline

# Global variable to hold the model pipeline
pipe = None

class InferenceHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            try:
                with open('index.html', 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_error(404, "File not found")
        elif self.path.startswith('/gallery/images'):
            self.handle_gallery_list()
        elif self.path.startswith('/gallery/image/'):
            self.handle_gallery_image()
        else:
            self.send_error(404, "Not Found")

    def handle_gallery_list(self):
        if not args.save_dir:
            self.send_error(400, "Gallery not available (save-dir not configured)")
            return

        try:
            # Parse query params for pagination
            query = self.path.split('?')[-1] if '?' in self.path else ''
            params = dict(qc.split('=') for qc in query.split('&') if '=' in qc)
            limit = int(params.get('limit', 20))
            offset = int(params.get('offset', 0))

            # Get list of images, sorted by time (newest first)
            files = glob.glob(os.path.join(args.save_dir, "*.png"))
            files.sort(key=os.path.getmtime, reverse=True)

            # Paginate
            paginated_files = files[offset:offset + limit]
            
            # Create response
            images = [os.path.basename(f) for f in paginated_files]
            response_data = {
                "images": images,
                "total": len(files),
                "has_more": (offset + limit) < len(files)
            }
            
            response_bytes = json.dumps(response_data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        except Exception as e:
            self.send_error(500, str(e))

    def handle_gallery_image(self):
        if not args.save_dir:
            self.send_error(404, "Not Found")
            return

        filename = self.path.split('/')[-1]
        # Basic security check to prevent directory traversal
        if '..' in filename or '/' in filename:
             self.send_error(403, "Forbidden")
             return

        filepath = os.path.join(args.save_dir, filename)
        if not os.path.exists(filepath):
            self.send_error(404, "Not Found")
            return

        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def do_POST(self):
        if self.path != '/generate':
            self.send_error(404, "Not Found")
            return

        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_error(400, "Bad Request: Missing body")
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body)
            prompt = data.get('prompt')
            
            if not prompt:
                self.send_error(400, "Bad Request: Missing 'prompt' field")
                return

            width = data.get('width', 1024)
            height = data.get('height', 1024)
            seed = data.get('seed', -1)

            # Validate resolution
            if width % 16 != 0 or height % 16 != 0:
                self.send_error(400, "Bad Request: Width and height must be divisible by 16")
                return

            # Prepare generator
            generator = None
            if seed != -1:
                generator = torch.Generator(device=pipe.device).manual_seed(seed)

            # Run inference
            # Hardcoded settings: num_inference_steps=8, guidance_scale=0.0
            image = pipe(
                prompt=prompt,
                num_inference_steps=8,
                height=height,
                width=width,
                guidance_scale=0.0,
                generator=generator
            ).images[0]

            # Convert image to PNG bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Save to disk if configured
            if args.save_dir:
                timestamp = int(time.time())
                filename = f"generated_{timestamp}_{seed}.png"
                filepath = os.path.join(args.save_dir, filename)
                try:
                    with open(filepath, 'wb') as f:
                        f.write(img_byte_arr)
                    print(f"Saved image to {filepath}")
                except Exception as e:
                    print(f"Failed to save image: {e}")

            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(img_byte_arr)))
            self.end_headers()
            self.wfile.write(img_byte_arr)

        except json.JSONDecodeError:
            self.send_error(400, "Bad Request: Invalid JSON")
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")

def main():
    global pipe, args
    parser = argparse.ArgumentParser(description='Z-Image-Turbo HTTP Server')
    parser.add_argument('--low-mem', action='store_true', help='Use CPU offload for low memory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on (default: 8000)')
    parser.add_argument('--save-dir', type=str, help='Directory to save generated images')
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Saving images to {args.save_dir}")

    print(f"Initializing model 'Tongyi-MAI/Z-Image-Turbo'...")
    
    # Load model
    pipe = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16
    )

    if args.low_mem:
        print("Low memory mode enabled: Using model CPU offload.")
        pipe.enable_model_cpu_offload()
    else:
        print(f"Moving model to {args.device}...")
        pipe.to(args.device)

    server_address = ('', args.port)
    httpd = http.server.HTTPServer(server_address, InferenceHandler)
    print(f"Server started on port {args.port}")
    httpd.serve_forever()

if __name__ == '__main__':
    main()
