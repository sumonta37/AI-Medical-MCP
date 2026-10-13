import traceback

from fastapi import (
    FastAPI,
    HTTPException
)

from pydantic import BaseModel


from agent_service import (
    run_medical_agent
)


from guardrails import (
    check_input_guardrails,
    check_output_guardrails
)


from hitl import (
    create_approval_request,
    get_approval_request,
    approve_request,
    reject_request
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(

    title="Medical AI Assistant API",

    description=(
        "Medical Agentic RAG with "
        "Guardrails, HITL and MCP"
    ),

    version="3.0.0"

)


# =========================================================
# REQUEST MODELS
# =========================================================

class QuestionRequest(BaseModel):

    question: str


class ApprovalRequest(BaseModel):

    approval_id: str


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "status":
            "running",

        "message": (

            "Medical AI Assistant "

            "with Guardrails, HITL and MCP"

        )

    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy"

    }


# =========================================================
# ASK
# =========================================================

@app.post("/ask")
async def ask_question(

    request:
        QuestionRequest

):

    question = (

        request.question.strip()

    )


    if not question:

        raise HTTPException(

            status_code=400,

            detail=(
                "Question cannot be empty."
            )

        )


    # =====================================================
    # INPUT GUARDRAIL
    # =====================================================

    guardrail = (

        check_input_guardrails(

            question

        )

    )


    # =====================================================
    # BLOCK
    # =====================================================

    if not guardrail.allowed:

        return {

            "status":
                "BLOCKED",

            "answer": (

                "I cannot process "

                "this request."

            ),

            "reason":
                guardrail.reason,

            "sources":
                []

        }


    # =====================================================
    # HUMAN-IN-THE-LOOP
    # =====================================================

    if guardrail.requires_human:

        approval = (

            create_approval_request(

                question=
                    question,

                reason=
                    guardrail.reason

            )

        )


        return {

            "status":
                "HUMAN_REVIEW",

            "approval_id":

                approval[
                    "approval_id"
                ],

            "answer": (

                "This request requires "

                "human review before "

                "processing."

            ),

            "message": (

                "This request requires "

                "human review before "

                "processing."

            ),

            "reason":
                guardrail.reason,

            "sources":
                []

        }


    # =====================================================
    # AGENT + MCP
    # =====================================================

    try:

        print()
        print(
            "=" * 80
        )

        print(
            "STARTING MEDICAL AGENT"
        )

        print(
            "=" * 80
        )

        print(
            f"Question: {question}"
        )


        # =================================================
        # ASYNC AGENT
        # =================================================

        result = (

            await run_medical_agent(

                question

            )

        )


        answer = (

            result.get(

                "answer",

                ""

            )

        )


        # =================================================
        # OUTPUT GUARDRAIL
        # =================================================

        output_guardrail = (

            check_output_guardrails(

                answer

            )

        )


        if not output_guardrail.allowed:

            return {

                "status":
                    "OUTPUT_BLOCKED",

                "answer": (

                    "The generated "

                    "response could not "

                    "pass the safety check."

                ),

                "reason":
                    output_guardrail.reason,

                "sources":
                    []

            }


        # =================================================
        # SUCCESS
        # =================================================

        return {

            "status":
                "COMPLETED",

            "answer":
                answer,

            "sources":

                result.get(

                    "sources",

                    []

                )

        }


    except Exception as error:

        print()
        print(
            "=" * 80
        )

        print(
            "AGENT / MCP ERROR"
        )

        print(
            "=" * 80
        )

        print(
            f"Exception type: "
            f"{type(error)}"
        )

        print(
            f"Exception: "
            f"{repr(error)}"
        )

        print()
        print(
            "FULL TRACEBACK:"
        )

        print(
            "=" * 80
        )


        traceback.print_exc()


        # =================================================
        # EXCEPTION GROUP
        # =================================================

        if isinstance(

            error,

            ExceptionGroup

        ):

            print()
            print(
                "=" * 80
            )

            print(
                "EXCEPTION GROUP DETAILS"
            )

            print(
                "=" * 80
            )


            for index, sub_error in enumerate(

                error.exceptions,

                start=1

            ):

                print()

                print(
                    f"SUB-EXCEPTION {index}"
                )

                print(
                    "-" * 60
                )

                print(
                    f"Type: "
                    f"{type(sub_error)}"
                )

                print(
                    f"Error: "
                    f"{repr(sub_error)}"
                )

                print()

                traceback.print_exception(

                    sub_error

                )


        raise HTTPException(

            status_code=500,

            detail=str(error)

        )


# =========================================================
# GET APPROVAL
# =========================================================

@app.get(
    "/approval/{approval_id}"
)
def get_approval(

    approval_id:
        str

):

    request = (

        get_approval_request(

            approval_id

        )

    )


    if not request:

        raise HTTPException(

            status_code=404,

            detail=(

                "Approval request "

                "not found."

            )

        )


    return request


# =========================================================
# APPROVE
# =========================================================

@app.post(
    "/approval/approve"
)
async def approve(

    request:
        ApprovalRequest

):

    approval = (

        approve_request(

            request.approval_id

        )

    )


    if not approval:

        raise HTTPException(

            status_code=404,

            detail=(

                "Approval request "

                "not found."

            )

        )


    try:

        print()
        print(
            "=" * 80
        )

        print(
            "HUMAN APPROVED REQUEST"
        )

        print(
            "=" * 80
        )

        print(
            f"Question: "
            f"{approval['question']}"
        )


        # =================================================
        # HUMAN APPROVED
        #
        # HITL
        #   ↓
        # Agent
        #   ↓
        # MCP
        #   ↓
        # RAG
        # =================================================

        result = (

            await run_medical_agent(

                approval[
                    "question"
                ]

            )

        )


        answer = (

            result.get(

                "answer",

                ""

            )

        )


        # =================================================
        # OUTPUT GUARDRAIL
        # =================================================

        output_guardrail = (

            check_output_guardrails(

                answer

            )

        )


        if not output_guardrail.allowed:

            return {

                "status":
                    "OUTPUT_BLOCKED",

                "answer": (

                    "The response "

                    "failed the output "

                    "safety check."

                ),

                "reason":
                    output_guardrail.reason,

                "sources":
                    []

            }


        # =================================================
        # SUCCESS
        # =================================================

        return {

            "status":
                "APPROVED_AND_COMPLETED",

            "answer":
                answer,

            "message":
                "Request approved and processed.",

            "sources":

                result.get(

                    "sources",

                    []

                )

        }


    except Exception as error:

        print()
        print(
            "=" * 80
        )

        print(
            "HITL APPROVAL / "
            "AGENT / MCP ERROR"
        )

        print(
            "=" * 80
        )

        print(
            f"Exception type: "
            f"{type(error)}"
        )

        print(
            f"Exception: "
            f"{repr(error)}"
        )

        print()

        traceback.print_exc()


        if isinstance(

            error,

            ExceptionGroup

        ):

            print()
            print(
                "EXCEPTION GROUP DETAILS"
            )


            for index, sub_error in enumerate(

                error.exceptions,

                start=1

            ):

                print()

                print(
                    f"SUB-EXCEPTION {index}"
                )

                print(
                    f"Type: "
                    f"{type(sub_error)}"
                )

                print(
                    f"Error: "
                    f"{repr(sub_error)}"
                )

                traceback.print_exception(

                    sub_error

                )


        raise HTTPException(

            status_code=500,

            detail=str(error)

        )


# =========================================================
# REJECT
# =========================================================

@app.post(
    "/approval/reject"
)
def reject(

    request:
        ApprovalRequest

):

    approval = (

        reject_request(

            request.approval_id

        )

    )


    if not approval:

        raise HTTPException(

            status_code=404,

            detail=(

                "Approval request "

                "not found."

            )

        )


    return {

        "status":
            "REJECTED",

        "answer": (

            "The request was "

            "rejected by the "

            "human reviewer."

        ),

        "message": (

            "The request was "

            "rejected by the "

            "human reviewer."

        ),

        "sources":
            []

    }