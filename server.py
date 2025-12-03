import http.server
import json
import argparse
import io
import torch
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
        else:
            self.send_error(404, "Not Found")

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

            # Run inference
            # Hardcoded settings: num_inference_steps=8, guidance_scale=0.0
            image = pipe(
                prompt=prompt,
                num_inference_steps=8,
                guidance_scale=0.0
            ).images[0]

            # Convert image to PNG bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

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
    global pipe
    parser = argparse.ArgumentParser(description='Z-Image-Turbo HTTP Server')
    parser.add_argument('--low-mem', action='store_true', help='Use CPU offload for low memory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on (default: 8000)')
    args = parser.parse_args()

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
