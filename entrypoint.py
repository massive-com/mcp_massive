#!/usr/bin/env python
"""
Backwards-compatible entrypoint script.
This script delegates to the main package CLI entry point.
"""

from mcp_massive import main

if __name__ == "__main__":
    main()
