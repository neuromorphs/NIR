"""Command-line interface for the NIR hub."""

import argparse
import json
import os
import sys

import nir
from . import upload, download
from .server import run_server


def main():
    """Main entry point for NIR hub CLI."""
    parser = argparse.ArgumentParser(description="NIR Model Hub CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a model to the hub")
    upload_parser.add_argument("nir_file", help="Path to NIR file")
    upload_parser.add_argument("--name", "-n", required=True, help="Model name")
    upload_parser.add_argument("--description", "-d", default="", help="Model description")
    upload_parser.add_argument("--tags", "-t", nargs="+", default=[], help="Tags for the model")
    upload_parser.add_argument("--framework", "-f", default="", help="Framework origin")
    upload_parser.add_argument("--platforms", "-p", nargs="+", default=[], 
                        help="Compatible platforms")
    upload_parser.add_argument("--url", default="http://localhost:8080", 
                        help="Hub server URL")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model from the hub")
    download_parser.add_argument("model_id", help="Model ID or name")
    download_parser.add_argument("--output", "-o", help="Output directory")
    download_parser.add_argument("--no-check", action="store_true", 
                          help="Skip compatibility check")
    download_parser.add_argument("--url", default="http://localhost:8080", 
                          help="Hub server URL")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for models")
    search_parser.add_argument("--tag", help="Filter by tag")
    search_parser.add_argument("--framework", help="Filter by framework")
    search_parser.add_argument("--platform", help="Filter by platform")
    search_parser.add_argument("--url", default="http://localhost:8080", 
                         help="Hub server URL")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the hub server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to listen on")
    server_parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080, avoid 5000 on macOS)")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.command == "upload":
        # Load the model
        try:
            graph = nir.read(args.nir_file)
        except Exception as e:
            print(f"Error loading NIR file: {e}", file=sys.stderr)
            return 1
        
        # Upload the model
        try:
            result = upload(
                graph=graph,
                model_name=args.name,
                description=args.description,
                tags=args.tags,
                framework_origin=args.framework,
                compatible_platforms=args.platforms,
                hub_url=args.url
            )
            print(f"Model uploaded successfully! ID: {result['model_id']}")
        except Exception as e:
            print(f"Error uploading model: {e}", file=sys.stderr)
            return 1
    
    elif args.command == "download":
        try:
            graph = download(
                model_id_or_name=args.model_id,
                output_dir=args.output,
                check_compatibility=not args.no_check,
                hub_url=args.url
            )
            print(f"Model downloaded successfully to {args.output if args.output else 'temporary directory'}")
        except Exception as e:
            print(f"Error downloading model: {e}", file=sys.stderr)
            return 1
    
    elif args.command == "search":
        import requests
        try:
            params = {}
            if args.tag:
                params["tag"] = args.tag
            if args.framework:
                params["framework"] = args.framework
            if args.platform:
                params["platform"] = args.platform
            
            response = requests.get(f"{args.url}/api/models/search", params=params)
            response.raise_for_status()
            results = response.json()
            
            if not results:
                print("No models found matching your criteria.")
            else:
                print(f"Found {len(results)} model(s):")
                for model in results:
                    tags = ", ".join(model.get("tags", []))
                    print(f"ID: {model['model_id']}")
                    print(f"Name: {model['model_name']}")
                    print(f"Description: {model.get('description', '')}")
                    print(f"Tags: {tags}")
                    print(f"NIR Version: {model.get('nir_version', '')}")
                    print("---")
        except Exception as e:
            print(f"Error searching models: {e}", file=sys.stderr)
            return 1
    
    elif args.command == "server":
        try:
            print(f"Starting NIR Hub server on {args.host}:{args.port}...")
            run_server(host=args.host, port=args.port, debug=args.debug)
        except Exception as e:
            print(f"Error running server: {e}", file=sys.stderr)
            return 1
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
