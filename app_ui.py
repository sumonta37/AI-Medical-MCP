import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="centered"
)


# =========================================================
# API Functions
# =========================================================

def call_api(question: str):

    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()


def approve_request(approval_id: str):

    response = requests.post(
        f"{API_URL}/approval/approve",
        json={
            "approval_id": approval_id
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()


def reject_request(approval_id: str):

    response = requests.post(
        f"{API_URL}/approval/reject",
        json={
            "approval_id": approval_id
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "pending_approval" not in st.session_state:

    st.session_state.pending_approval = None


# =========================================================
# Header
# =========================================================

st.title("🩺 Medical AI Assistant")

st.caption(
    "LlamaIndex + ChromaDB + Ollama Cloud + "
    "ReAct + Guardrails + HITL"
)


st.info(
    "This application provides information from the "
    "configured medical documents. It is not a substitute "
    "for professional medical advice."
)


# =========================================================
# Display Chat History
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# HUMAN-IN-THE-LOOP APPROVAL
# =========================================================

if st.session_state.pending_approval:

    approval = st.session_state.pending_approval

    st.divider()

    st.warning(
        "⚠️ Human approval is required before "
        "the agent can continue."
    )

    st.subheader(
        "🔐 Human Approval Required"
    )

    st.write(
        "**Requested action:**"
    )

    st.info(
        approval.get(
            "message",
            "The agent is requesting human approval."
        )
    )

    approval_id = approval.get(
        "approval_id"
    )

    if not approval_id:

        st.error(
            "Approval ID is missing from the API response."
        )

    else:

        col1, col2 = st.columns(2)


        # =================================================
        # APPROVE
        # =================================================

        with col1:

            if st.button(
                "✅ Approve",
                use_container_width=True,
                type="primary"
            ):

                try:

                    with st.spinner(
                        "Processing approved request..."
                    ):

                        result = approve_request(
                            approval_id
                        )


                    # -----------------------------------------
                    # Display result
                    # -----------------------------------------

                    answer = result.get(
                        "answer"
                    )

                    message = result.get(
                        "message",
                        "Request approved."
                    )


                    st.success(
                        message
                    )


                    if answer:

                        st.markdown(
                            "### 🤖 Agent Response"
                        )

                        st.markdown(
                            answer
                        )


                        # Save response
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )


                    # Clear approval
                    st.session_state.pending_approval = None

                    st.rerun()


                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to FastAPI. "
                        "Please make sure the backend is running."
                    )


                except requests.exceptions.Timeout:

                    st.error(
                        "The approval request timed out."
                    )


                except requests.exceptions.RequestException as error:

                    st.error(
                        f"Approval API error: {error}"
                    )


                except Exception as error:

                    st.error(
                        f"Unexpected approval error: {error}"
                    )


        # =================================================
        # REJECT
        # =================================================

        with col2:

            if st.button(
                "❌ Reject",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Rejecting request..."
                    ):

                        result = reject_request(
                            approval_id
                        )


                    st.warning(
                        result.get(
                            "message",
                            "Request rejected."
                        )
                    )


                    # Clear approval
                    st.session_state.pending_approval = None

                    st.rerun()


                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to FastAPI. "
                        "Please make sure the backend is running."
                    )


                except requests.exceptions.Timeout:

                    st.error(
                        "The rejection request timed out."
                    )


                except requests.exceptions.RequestException as error:

                    st.error(
                        f"Rejection API error: {error}"
                    )


                except Exception as error:

                    st.error(
                        f"Unexpected rejection error: {error}"
                    )


    st.divider()


# =========================================================
# Chat Input
# =========================================================

question = st.chat_input(
    "Ask your medical question..."
)


if question:

    # =====================================================
    # Don't allow another request while approval is pending
    # =====================================================

    if st.session_state.pending_approval:

        st.warning(
            "Please approve or reject the pending request "
            "before asking another question."
        )

        st.stop()


    # =====================================================
    # Save user message
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # =====================================================
    # Display user message
    # =====================================================

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # =====================================================
    # Call Agent
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Agent is thinking..."
        ):

            try:

                result = call_api(
                    question
                )


                # =================================================
                # CASE 1
                # Human approval required
                # =================================================

                if (
                    result.get("requires_approval", False)
                    or
                    result.get("status") == "HUMAN_REVIEW"
                    ):

                    approval = {

                        "approval_id": result.get(
                            "approval_id"
                        ),

                        "message": result.get(
                            "message",
                            "The agent requires human approval."
                        )
                    }


                    # ---------------------------------------------
                    # Save pending approval
                    # ---------------------------------------------

                    st.session_state.pending_approval = approval


                    # ---------------------------------------------
                    # Tell user
                    # ---------------------------------------------

                    st.warning(
                        "⚠️ This request requires human approval."
                    )


                    # ---------------------------------------------
                    # Rerun so approval buttons appear
                    # ---------------------------------------------

                    st.rerun()


                # =================================================
                # CASE 2
                # Normal answer
                # =================================================

                else:

                    answer = result.get(
                        "answer",
                        "No answer returned."
                    )


                    # ---------------------------------------------
                    # Display answer
                    # ---------------------------------------------

                    st.markdown(
                        answer
                    )


                    # ---------------------------------------------
                    # Retrieved sources
                    # ---------------------------------------------

                    sources = result.get(
                        "sources",
                        []
                    )


                    if sources:

                        with st.expander(
                            "📄 Retrieved Sources"
                        ):

                            for source in sources:

                                file_name = source.get(
                                    "file_name",
                                    "Unknown document"
                                )


                                st.write(
                                    f"**{file_name}**"
                                )


                                if "similarity" in source:

                                    st.write(
                                        "Similarity:",
                                        source["similarity"]
                                    )


                    # ---------------------------------------------
                    # Save assistant response
                    # ---------------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


            # =====================================================
            # Connection Error
            # =====================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI.\n\n"
                    "Please make sure the backend is running:"
                )

                st.code(
                    "python -m uvicorn api:app --reload --port 8000"
                )


            # =====================================================
            # Timeout
            # =====================================================

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The request timed out."
                )


            # =====================================================
            # HTTP/API Error
            # =====================================================

            except requests.exceptions.HTTPError as error:

                st.error(
                    f"❌ FastAPI returned an error: {error}"
                )


            except requests.exceptions.RequestException as error:

                st.error(
                    f"❌ API request failed: {error}"
                )


            # =====================================================
            # Unexpected Error
            # =====================================================

            except Exception as error:

                st.error(
                    f"❌ Unexpected error: {error}"
                )