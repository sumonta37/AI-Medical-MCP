import asyncio
import json
import traceback

from mcp import ClientSession

from mcp.client.streamable_http import (
    streamable_http_client
)


# =========================================================
# MCP SERVER
# =========================================================

MCP_SERVER_URL = (
    "http://127.0.0.1:8001/mcp"
)


# =========================================================
# CALL MEDICAL DOCUMENT SEARCH
# =========================================================

async def call_medical_document_search(
    query: str
):

    print()
    print("=" * 60)
    print("MCP CLIENT")
    print("=" * 60)

    print(
        f"Connecting to MCP server: "
        f"{MCP_SERVER_URL}"
    )


    try:

        # =================================================
        # STREAMABLE HTTP
        #
        # IMPORTANT:
        # Your MCP SDK returns TWO values.
        # =================================================

        async with streamable_http_client(
            MCP_SERVER_URL
        ) as (

            read_stream,

            write_stream

        ):

            print(
                "MCP HTTP connection established."
            )


            # =================================================
            # MCP SESSION
            # =================================================

            async with ClientSession(

                read_stream,

                write_stream

            ) as session:

                print(
                    "Initializing MCP session..."
                )


                # =================================================
                # INITIALIZE
                # =================================================

                await session.initialize()

                print(
                    "MCP session initialized."
                )


                # =================================================
                # LIST TOOLS
                # =================================================

                tools_result = (
                    await session.list_tools()
                )


                tools = [

                    tool.name

                    for tool
                    in tools_result.tools

                ]


                print()
                print(
                    "========== MCP TOOLS =========="
                )

                print(
                    tools
                )

                print(
                    "==============================="
                )


                # =================================================
                # CHECK TOOL
                # =================================================

                if (
                    "medical_document_search"
                    not in tools
                ):

                    raise RuntimeError(

                        "MCP tool "
                        "'medical_document_search' "
                        "was not found."

                    )


                # =================================================
                # CALL MCP TOOL
                # =================================================

                print()
                print(
                    "========== MCP REQUEST =========="
                )

                print(
                    "Tool  : medical_document_search"
                )

                print(
                    f"Query : {query}"
                )

                print(
                    "================================="
                )


                result = await session.call_tool(

                    "medical_document_search",

                    arguments={

                        "query":
                            query

                    }

                )


                print()
                print(
                    "MCP tool execution completed."
                )


                # =================================================
                # EXTRACT RESULT
                # =================================================

                return extract_mcp_result(
                    result
                )


    except Exception as error:

        print()
        print("=" * 60)
        print("MCP CLIENT ERROR")
        print("=" * 60)

        print(
            f"Exception type: {type(error)}"
        )

        print(
            f"Exception: {repr(error)}"
        )

        print()

        traceback.print_exc()

        raise


# =========================================================
# EXTRACT MCP RESULT
# =========================================================

def extract_mcp_result(
    result
):

    print()
    print(
        "========== MCP RAW RESULT =========="
    )

    print(
        result
    )

    print(
        "===================================="
    )


    # =================================================
    # STRUCTURED CONTENT
    # =================================================

    structured_content = getattr(

        result,

        "structuredContent",

        None

    )


    if structured_content:

        return structured_content


    # =================================================
    # TEXT CONTENT
    # =================================================

    content = getattr(

        result,

        "content",

        None

    )


    if content:

        for item in content:

            text = getattr(

                item,

                "text",

                None

            )


            if text:

                try:

                    return json.loads(
                        text
                    )

                except json.JSONDecodeError:

                    return {

                        "found":
                            True,

                        "context":
                            text,

                        "sources":
                            []

                    }


    # =================================================
    # EMPTY RESULT
    # =================================================

    return {

        "found":
            False,

        "context":
            "",

        "sources":
            []

    }


# =========================================================
# STANDALONE MCP TEST
# =========================================================

async def test_mcp():

    result = (
        await call_medical_document_search(

            "symptoms of diabetes"

        )
    )


    print()
    print("=" * 60)
    print("FINAL MCP RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print(
        "=" * 60
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        test_mcp()
    )