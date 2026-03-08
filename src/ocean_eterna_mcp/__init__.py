"""Ocean Eterna MCP Server -- BM25 search engine as MCP tools."""

from importlib.metadata import version as _v

try:
    __version__ = _v("ocean-eterna-mcp")
except Exception:
    __version__ = "4.3.0"


def main():
    """entry point for the ocean-eterna-mcp CLI command."""
    import sys
    import os

    # add parent dir so doc_processor can be found if installed alongside
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # import and run the server
    from ocean_eterna_mcp.server import mcp
    mcp.run(transport="stdio")
