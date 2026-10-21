"""
Command-line interface (CLI) for running the SafeStep API server or quick model inference.
"""

import argparse
import json
import sys

from api.config import default_api_config
from api.security import fetch_external_image, fetch_external_text
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
    text_parser.add_argument("text", nargs="?", default=None, help="Text string to evaluate")
    text_parser.add_argument("--url", default=None, help="External URL to fetch text content from securely")
    text_parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (0.0 to 1.0)")
    text_parser.add_argument("--features", action="store_true", help="Include extracted stylistic heuristic features")

    # Command: predict-phishing
    phish_parser = subparsers.add_parser("predict-phishing", help="Run the phishing escalation pipeline directly")
    phish_parser.add_argument("text", nargs="?", default=None, help="Email body, URL, or page text to evaluate")
    phish_parser.add_argument("--url", default=None, help="External URL to fetch content from securely")
    phish_parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (0.0 to 1.0)")
    phish_parser.add_argument(
        "--max-stage", default="warhammer", choices=["attack", "beast", "warhammer"],
        help="Stop escalating after this stage",
    )

    # Command: predict-image
    img_parser = subparsers.add_parser("predict-image", help="Run JAW image detection directly")
    img_parser.add_argument("image_path", nargs="?", default=None, help="Path to image file")
    img_parser.add_argument("--url", default=None, help="External URL to fetch image from securely")
    img_parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (0.0 to 1.0)")

    # Command: predict-unified
    unified_parser = subparsers.add_parser("predict-unified", help="Run multimodal synthetic risk analysis")
    unified_parser.add_argument("--text", default=None, help="Text content to evaluate")
    unified_parser.add_argument("--text-url", default=None, help="External URL to fetch text from")
    unified_parser.add_argument("--image", default=None, help="Path to image to evaluate")
    unified_parser.add_argument("--image-url", default=None, help="External URL to fetch image from")
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
        target = args.text
        if not target and args.url:
            target = fetch_external_text(args.url)
        elif not target:
            print("Error: Must provide text argument or --url.", file=sys.stderr)
            sys.exit(1)
        res = service.predict_text(target, threshold=args.threshold, include_features=args.features)
        print(json.dumps(res.model_dump(), indent=2))

    elif args.command == "predict-phishing":
        target = args.text
        if not target and args.url:
            fetched = fetch_external_text(args.url)
            target = f"{args.url}\n\n{fetched}"
        elif not target:
            print("Error: Must provide text argument or --url.", file=sys.stderr)
            sys.exit(1)
        res = service.predict_phishing(target, threshold=args.threshold, max_stage=args.max_stage)
        print(json.dumps(res.model_dump(), indent=2))

    elif args.command == "predict-image":
        target = args.image_path
        if not target and args.url:
            target = fetch_external_image(args.url)
        elif not target:
            print("Error: Must provide image_path argument or --url.", file=sys.stderr)
            sys.exit(1)
        res = service.predict_image(target, threshold=args.threshold)
        print(json.dumps(res.model_dump(), indent=2))

    elif args.command == "predict-unified":
        text_target = args.text
        if not text_target and args.text_url:
            text_target = fetch_external_text(args.text_url)

        image_target = args.image
        if not image_target and args.image_url:
            image_target = fetch_external_image(args.image_url)

        if not text_target and not image_target:
            print("Error: At least one of --text/--text-url or --image/--image-url must be provided.", file=sys.stderr)
            sys.exit(1)
        res = service.predict_unified(
            text=text_target,
            image=image_target,
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
