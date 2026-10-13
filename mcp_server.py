from mcp.server.fastmcp import FastMCP

from rag_service import search_medical_documents

import uvicorn


# ==========================================================
# MCP SERVER
# ==========================================================

mcp = FastMCP(
    "Medical AI MCP Server"
)


# ==========================================================
# MCP TOOL
# ==========================================================

@mcp.tool()
def medical_document_search(query: str) -> dict:
    """
    Search the indexed medical documents.
    """

    if not query or not query.strip():

        return {
            "found": False,
            "context": "",
            "sources": [],
            "error": "Query cannot be empty."
        }

    try:

        result = search_medical_documents(
            query.strip()
        )

        return result

    except Exception as e:

        return {
            "found": False,
            "context": "",
            "sources": [],
            "error": str(e)
        }


# ==========================================================
# MCP HTTP APPLICATION
# ==========================================================

app = mcp.streamable_http_app()


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MEDICAL AI MCP SERVER")
    print("=" * 60)

    print(
        "MCP endpoint:"
        " http://127.0.0.1:8001/mcp"
    )

    print("=" * 60)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001
    )