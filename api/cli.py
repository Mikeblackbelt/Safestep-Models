"""
Command-line interface (CLI) for running the SafeStep API server or quick model inference.
"""

import argparse
import json
import sys

from api.config import default_api_config
from api.service import service


def main():
    parser = argparse.ArgumentParser(
        prog="safestep-api",
        description="SafeStep AI Detection API and CLI Runner",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI HTTP server")
    serve_parser.add_argument("--host", default=default_api_config.host, help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=default_api_config.port, help="Bind port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")

    # Command: predict-text
    text_parser = subparsers.add_parser("predict-text", help="Run CART text detection directly")
    text_parser.add_argument("text", help="Text string to evaluate")
    text_parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (0.0 to 1.0)")
    text_parser.add_argument("--features", action="store_true", help="Include extracted stylistic heuristic features")

    # Command: predict-image
    img_parser = subparsers.add_parser("predict-image", help="Run JAW image detection directly")
    img_parser.add_argument("image_path", help="Path to image file")
    img_parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (0.0 to 1.0)")

    # Command: predict-unified
    unified_parser = subparsers.add_parser("predict-unified", help="Run multimodal synthetic risk analysis")
    unified_parser.add_argument("--text", default=None, help="Text content to evaluate")
    unified_parser.add_argument("--image", default=None, help="Path to image to evaluate")
    unified_parser.add_argument("--text-threshold", type=float, default=None, help="Custom text threshold (0.0 to 1.0)")
    unified_parser.add_argument("--image-threshold", type=float, default=None, help="Custom image threshold (0.0 to 1.0)")

    # Command: status
    subparsers.add_parser("status", help="Display status of registered models and checkpoints")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        print(f"Starting SafeStep API server on {args.host}:{args.port}...")
        print(f"Interactive API documentation available at http://{args.host}:{args.port}/docs")
        uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command == "predict-text":
        res = service.predict_text(args.text, threshold=args.threshold, include_features=args.features)
        print(json.dumps(res.model_dump(), indent=2))

    elif args.command == "predict-image":
        res = service.predict_image(args.image_path, threshold=args.threshold)
        print(json.dumps(res.model_dump(), indent=2))

    elif args.command == "predict-unified":
        if not args.text and not args.image:
            print("Error: At least one of --text or --image must be provided.", file=sys.stderr)
            sys.exit(1)
        res = service.predict_unified(
            text=args.text,
            image=args.image,
            text_threshold=args.text_threshold,
            image_threshold=args.image_threshold,
        )
        print(json.dumps(res.model_dump(), indent=2))

    elif args.command == "status":
        statuses = [s.model_dump() for s in service.registry.get_status()]
        print(json.dumps(statuses, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
