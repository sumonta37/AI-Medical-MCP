import uuid
from datetime import datetime


# ---------------------------------------------------------
# Simple in-memory approval store
#
# Production:
# Replace this with Redis/PostgreSQL/etc.
# ---------------------------------------------------------

pending_requests = {}


# ---------------------------------------------------------
# Create approval request
# ---------------------------------------------------------

def create_approval_request(
    question: str,
    reason: str
):

    approval_id = str(uuid.uuid4())

    pending_requests[approval_id] = {

        "approval_id": approval_id,

        "question": question,

        "reason": reason,

        "status": "PENDING",

        "created_at": datetime.utcnow().isoformat(),

    }

    return pending_requests[approval_id]


# ---------------------------------------------------------
# Get request
# ---------------------------------------------------------

def get_approval_request(approval_id: str):

    return pending_requests.get(approval_id)


# ---------------------------------------------------------
# Approve
# ---------------------------------------------------------

def approve_request(approval_id: str):

    request = pending_requests.get(approval_id)

    if not request:
        return None

    request["status"] = "APPROVED"

    return request


# ---------------------------------------------------------
# Reject
# ---------------------------------------------------------

def reject_request(approval_id: str):

    request = pending_requests.get(approval_id)

    if not request:
        return None

    request["status"] = "REJECTED"

    return request