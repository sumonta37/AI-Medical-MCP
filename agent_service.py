import json
import re
import os

from dotenv import load_dotenv
from ollama import Client

from mcp_client import (
    call_medical_document_search
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# OLLAMA
# =========================================================

OLLAMA_API_KEY = os.getenv(
    "OLLAMA_API_KEY"
)

if not OLLAMA_API_KEY:

    raise ValueError(
        "OLLAMA_API_KEY is missing."
    )


ollama_client = Client(

    host="https://ollama.com",

    headers={

        "Authorization":
            f"Bearer {OLLAMA_API_KEY}"

    }

)


# =========================================================
# SYSTEM PROMPT
# =========================================================

REACT_SYSTEM_PROMPT = """

You are a Medical AI Agent.

You have access to exactly ONE MCP tool:

medical_document_search

The MCP tool searches the indexed medical documents.

============================================================
AGENT WORKFLOW
============================================================

You MUST follow a two-step process.

============================================================
STEP 1 - TOOL
============================================================

For a medical question, first use the MCP tool.

Return EXACTLY ONE JSON object:

{
    "type": "tool",
    "tool": "medical_document_search",
    "query": "your search query"
}

STOP immediately after returning this JSON.

DO NOT generate a final answer in STEP 1.

DO NOT return another JSON object.

DO NOT return TOOL and FINAL together.

============================================================
STEP 2 - FINAL
============================================================

After the MCP tool result is provided, generate the
final answer using the retrieved information.

Return EXACTLY ONE JSON object:

{
    "type": "final",
    "answer": "your answer"
}

STOP immediately.

============================================================
MEDICAL SAFETY
============================================================

1. Do not invent medical information.

2. Use the MCP tool result as the source of
   medical-document information.

3. Do not diagnose patients.

4. Do not independently prescribe medication.

5. Do not provide individualized dosage recommendations.

6. If the retrieved documents do not contain enough
   information, clearly say so.

7. Keep the answer clear and concise.

8. Return valid JSON only.

9. Return EXACTLY ONE JSON object per response.

10. NEVER return TOOL and FINAL in the same response.

"""


# =========================================================
# ROBUST JSON PARSER
# =========================================================

def parse_agent_json(
    content: str
):
    """
    Parse the LLM response.

    The model may occasionally return multiple JSON
    objects in one response.

    If both TOOL and FINAL are returned, TOOL gets
    priority because the MCP retrieval must happen
    before the final grounded response.
    """

    content = content.strip()


    # =====================================================
    # REMOVE MARKDOWN CODE FENCES
    # =====================================================

    content = re.sub(

        r"```json",

        "",

        content,

        flags=re.IGNORECASE

    )


    content = re.sub(

        r"```",

        "",

        content

    )


    content = content.strip()


    # =====================================================
    # PARSE MULTIPLE JSON OBJECTS
    # =====================================================

    decoder = json.JSONDecoder()

    objects = []

    position = 0


    while position < len(content):

        # -------------------------------------------------
        # Skip whitespace
        # -------------------------------------------------

        while (

            position < len(content)

            and content[position].isspace()

        ):

            position += 1


        if position >= len(content):

            break


        # -------------------------------------------------
        # Decode one JSON object
        # -------------------------------------------------

        try:

            obj, next_position = (
                decoder.raw_decode(

                    content,

                    position

                )
            )


            objects.append(
                obj
            )


            position = (
                next_position
            )


        except json.JSONDecodeError as error:

            raise ValueError(

                "Unable to parse Agent response "
                "as JSON.\n\n"

                "Raw Agent response:\n"

                f"{content}"

            ) from error


    # =====================================================
    # NO OBJECT
    # =====================================================

    if not objects:

        raise ValueError(
            "Agent returned no JSON object."
        )


    # =====================================================
    # TOOL HAS PRIORITY
    # =====================================================

    for obj in objects:

        if (

            isinstance(
                obj,
                dict
            )

            and

            obj.get("type") == "tool"

        ):

            return obj


    # =====================================================
    # FINAL
    # =====================================================

    for obj in objects:

        if (

            isinstance(
                obj,
                dict
            )

            and

            obj.get("type") == "final"

        ):

            return obj


    # =====================================================
    # INVALID RESPONSE
    # =====================================================

    raise ValueError(

        "Agent returned JSON, but no valid "
        "'tool' or 'final' decision was found.\n\n"

        f"Parsed objects: {objects}"

    )


# =========================================================
# MCP TOOL EXECUTION
# =========================================================

async def execute_mcp_tool(

    tool_name: str,

    query: str

):

    if (

        tool_name
        != "medical_document_search"

    ):

        raise ValueError(

            f"Unknown MCP tool: "
            f"{tool_name}"

        )


    result = (

        await call_medical_document_search(

            query

        )

    )


    return result


# =========================================================
# MEDICAL AGENT
# =========================================================

async def run_medical_agent(

    question: str

):

    if (

        not question

        or

        not question.strip()

    ):

        raise ValueError(

            "Question cannot be empty."

        )


    # =====================================================
    # AGENT MESSAGES
    # =====================================================

    messages = [

        {

            "role":
                "system",

            "content":
                REACT_SYSTEM_PROMPT

        },

        {

            "role":
                "user",

            "content":
                question.strip()

        }

    ]


    # =====================================================
    # SOURCE TRACKING
    # =====================================================

    last_sources = []


    # =====================================================
    # AGENT LOOP
    # =====================================================

    for step in range(5):

        print()
        print(
            "=" * 60
        )

        print(
            f"AGENT STEP {step + 1}"
        )

        print(
            "=" * 60
        )


        # =================================================
        # CALL LLM
        # =================================================

        response = ollama_client.chat(

            model="gpt-oss:120b",

            messages=messages

        )


        content = (

            response[
                "message"
            ][
                "content"
            ]

        )


        print()
        print(
            "========== AGENT DECISION =========="
        )

        print(
            content
        )

        print(
            "===================================="
        )


        # =================================================
        # PARSE AGENT RESPONSE
        # =================================================

        decision = parse_agent_json(
            content
        )


        decision_type = (
            decision.get(
                "type"
            )
        )


        # =================================================
        # TOOL DECISION
        # =================================================

        if decision_type == "tool":

            tool_name = (
                decision.get(
                    "tool"
                )
            )


            query = (
                decision.get(
                    "query"
                )
            )


            if not tool_name:

                raise ValueError(
                    "Tool name missing."
                )


            if not query:

                raise ValueError(
                    "Tool query missing."
                )


            print()
            print(
                "========== MCP REQUEST =========="
            )

            print(
                f"Tool  : {tool_name}"
            )

            print(
                f"Query : {query}"
            )

            print(
                "================================="
            )


            # =================================================
            # EXECUTE MCP
            # =================================================

            try:

                tool_result = (

                    await execute_mcp_tool(

                        tool_name,

                        query

                    )

                )


            except Exception as error:

                print()
                print(
                    "=" * 60
                )

                print(
                    "MCP ERROR INSIDE AGENT"
                )

                print(
                    "=" * 60
                )

                print(
                    f"Type: {type(error)}"
                )

                print(
                    f"Error: {repr(error)}"
                )

                raise


            # =================================================
            # MCP RESULT
            # =================================================

            print()
            print(
                "========== MCP RESULT =========="
            )

            print(

                json.dumps(

                    tool_result,

                    indent=2

                )

            )

            print(
                "================================"
            )


            # =================================================
            # SAVE SOURCES
            # =================================================

            last_sources = (

                tool_result.get(

                    "sources",

                    []

                )

            )


            # =================================================
            # ADD AGENT DECISION
            # =================================================

            messages.append(

                {

                    "role":
                        "assistant",

                    "content":
                        content

                }

            )


            # =================================================
            # ADD MCP OBSERVATION
            # =================================================

            messages.append(

                {

                    "role":
                        "user",

                    "content":

                        (

                            "MCP TOOL RESULT:\n\n"

                            +

                            json.dumps(

                                tool_result,

                                indent=2

                            )

                            +

                            """



Now generate the final answer.

Use the information returned
by the MCP tool.

Do not invent medical information.

Do not diagnose the patient.

Return EXACTLY ONE JSON object:

{
    "type": "final",
    "answer": "..."
}

Do not return a tool request.

"""

                        )

                }

            )


            # =================================================
            # NEXT AGENT STEP
            # =================================================

            continue


        # =================================================
        # FINAL DECISION
        # =================================================

        if decision_type == "final":

            answer = (

                decision.get(

                    "answer",

                    ""

                )

            )


            if not answer:

                raise ValueError(

                    "Agent returned "
                    "empty answer."

                )


            return {

                "answer":
                    answer,

                "sources":
                    last_sources

            }


        # =================================================
        # UNKNOWN DECISION
        # =================================================

        raise ValueError(

            "Unknown agent response type: "

            f"{decision_type}"

        )


    # =====================================================
    # MAX STEPS
    # =====================================================

    raise RuntimeError(

        "Agent exceeded maximum steps."

    )