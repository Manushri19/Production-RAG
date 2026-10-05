"""
Serve Command - Start the Meridian HTTP API.
"""

import sys


def cmd_serve(args):
    """Start the HTTP API server."""
    try:
        import uvicorn
        from meridian.api.app import create_app
    except ImportError:
        print("Error: FastAPI/uvicorn not installed. Run: pip install -e '.[all]'")
        sys.exit(1)

    app = create_app()

    print(f"Starting Meridian API on {args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(app, host=args.host, port=args.port)
