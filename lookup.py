from langchain.tools import tool
import os
from typing import Annotated, Sequence, TypedDict
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from typing import List, Literal
from typing_extensions import TypedDict
from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState


import os
from dotenv import load_dotenv

load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "azure_openai:gpt-4.1",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)


USER_DATABASE = {
    "2871513": {
        "employee_name": "Arun",
        "employee_id": "2871513",
        "ultimatix_login_last": "12 hours ago",
        "domain_account_locked": {"state": False},
        "ultimatix_account_locked": {
            "state": True,
            "reason": "Wrong password attempts more than 2 times",
        },
        "domain_region": "India",
        "asset_id": None,
        "email": "arun.s@tcs.com",
        "bgv_status": "Pending",
        "project": "Bench",
        "grade": "Y",
    },
    "2987452": {
        "employee_name": "Priya",
        "employee_id": "2987452",
        "ultimatix_login_last": "2 days ago",
        "domain_account_locked": {"state": True, "reason": "Password expired"},
        "ultimatix_account_locked": {"state": False},
        "domain_region": "India",
        "asset_id": "AS12345",
        "email": "priya.k@tcs.com",
        "bgv_status": "Completed",
        "project": "Retail Transformation",
        "grade": "A1",
    },
    "2764019": {
        "employee_name": "Rahul",
        "employee_id": "2764019",
        "ultimatix_login_last": "Just now",
        "domain_account_locked": {"state": False},
        "ultimatix_account_locked": {"state": False},
        "domain_region": "Europe",
        "asset_id": "DE99881",
        "email": "rahul.m@tcs.com",
        "bgv_status": "Completed",
        "project": "Banking L2 Support",
        "grade": "A2",
    },
    "2890077": {
        "employee_name": "Sneha",
        "employee_id": "2890077",
        "ultimatix_login_last": "5 hours ago",
        "domain_account_locked": {
            "state": True,
            "reason": "Multiple wrong login attempts",
        },
        "ultimatix_account_locked": {
            "state": True,
            "reason": "Authenticator not registered",
        },
        "domain_region": "USA",
        "asset_id": None,
        "email": "sneha.p@tcs.com",
        "bgv_status": "Initiated",
        "project": "Onboarding",
        "grade": "Y",
    },
}


class CustomState(AgentState):
    employee_id: str


@tool
def get_employee_details(runtime: ToolRuntime[None, CustomState]) -> str:
    """
    Fetch detailed onboarding and account information for a specific TCS employee using their employee ID.

    This tool retrieves:
    - Ultimatix and Domain account lock status
    - Email and work region
    - Assigned project and employee grade
    - Background Verification (BGV) progress
    - Last Ultimatix login time
    - Asset assignment details

    Use this tool whenever the user asks anything about:
    - Account issues (locked/active status)
    - Email or region details
    - Assigned project or asset
    - BGV status or onboarding progress

    Returns a formatted employee profile summary if the ID is found,
    otherwise returns "Employee not found in system."
    """

    print("\nTool has been called\n")
    employee_id = runtime.state.get("employee_id")
    print(employee_id)

    if not employee_id or employee_id not in USER_DATABASE:
        return "Employee not found in system."

    employee = USER_DATABASE[employee_id]
    # Build detailed employee context
    account_status = []

    if employee["ultimatix_account_locked"]["state"]:
        account_status.append(
            f"🔒 Ultimatix Account LOCKED: {employee['ultimatix_account_locked']['reason']}"
        )
    else:
        account_status.append("✅ Ultimatix Account: Active")

    if employee["domain_account_locked"]["state"]:
        account_status.append(
            f"🔒 Domain Account LOCKED: {employee['domain_account_locked'].get('reason', 'Unknown')}"
        )
    else:
        account_status.append("✅ Domain Account: Active")

    return employee


# tools=[get_employee_details_test]


class replytype(TypedDict):
    answer: bool  |None  = Field(
        ...,
        description="Please provide the reason behind your decision, whether it is a yes or a no."
    )
    reason: str = Field(
        ...,
        description="Lookup tool that verifies if a question is answerable from the employee JSON fields",
    )


lookup_agent = create_agent(
    llm,
    tools=[get_employee_details],
    system_prompt="""
You are a Knowledge-Base Lookup Agent whose sole job is to check whether a user's natural-language question can be answered from the employee JSON and to return a tiny JSON reply with exactly two fields: "answer" and "reason", matching the `replytype` schema.

Always use the `get_employee_details` tool to fetch the employee record whenever an employee identifier (employee id or other explicit identifier present in the user input or agent state) is available. Do not attempt to answer without first fetching the record when an id is available.



====================================================================
EMPLOYEE JSON — FIELD MEANINGS (IMPORTANT FOR CORRECT MAPPING)
====================================================================

The employee record may include these fields. Use these definitions to semantically map user questions:

- employee_name → The employee's full name.
- employee_id → The employee's unique identifier.
- email → The employee’s official TCS email address.
- grade → The employee’s grade/band (e.g., Y, A1, A2).
- project → The project or assignment the employee is currently mapped to (e.g., Bench, Retail Transformation).
- domain_region → The geographical region of the employee from the AD/Domain perspective.
- asset_id → The laptop/desktop or assigned asset identifier (if any).
- bgv_status → Background verification status (e.g., Pending, Completed, Initiated).
- ultimatix_login_last → When the employee last logged in to Ultimatix.
- domain_account_locked.state → Whether the employee’s domain account is locked (true/false).
- domain_account_locked.reason → Why the domain account is locked (if locked).
- ultimatix_account_locked.state → Whether the employee’s Ultimatix account is locked (true/false).
- ultimatix_account_locked.reason → Why the Ultimatix account is locked (if locked).


OUTPUT REQUIREMENTS (MANDATORY)
- Return strictly valid JSON only. No prose outside JSON.
- The JSON MUST contain exactly two keys: 
  1) "answer" — set to `true`, `false`, or `null` (JSON `null`) only.  
  2) "reason" — a single string containing the human-readable reply or explanation.
- No extra keys, no arrays/lists, no raw JSON dumps of the record, and no internal field names or JSON keys in the "reason".
- Do not include any additional commentary, diagnostics, or metadata.

INTERPRETATION RULES (HOW TO SET "answer" AND WHAT TO PUT IN "reason")

1) GENERAL PERSONAL / FACTUAL QUESTIONS (examples: "what is my name?", "what is my email?", "what is my asset id?", "what is my grade?", "what project am I on?", "where am I based?", "when was my last ultimatix login?")
- If the requested factual item exists in the employee JSON and provides the complete direct answer:
  - Set `"answer": null`
  - Put the direct human-readable answer in `"reason"`. Example forms:
    - "Your name is xxx."
    - "Your registered email is xxx@zzz.com."
    - "Your asset id is AS12345."
  - Use natural language only; do not show JSON key names or raw JSON.

2) ISSUE / TROUBLESHOOTING QUESTIONS (examples: "is my Ultimatix locked?", "why can't I login?", "is my domain account locked?", "why was my account locked?")
- If the employee JSON contains the specific issue-related data required to fully answer the question (e.g., lock state booleans, lock reasons, last login timestamp):
  - Set `"answer": true`
  - Write an explicit, human-readable explanation of the issue in `"reason"`. Include relevant values from the JSON in natural-language sentences (for example: lock status, lock reason text, and last-login time). Do not reveal internal field names—phrase them naturally:
    - Example: "Your Ultimatix account is locked because of wrong password attempts more than 2 times; last successful/attempted login was 12 hours ago."
    - Example: "Your domain account is locked due to password expired."
  - If multiple related issue fields exist (e.g., both domain and Ultimatix locked), include both issues clearly and concisely in the same `reason` string.

3) PARTIAL / INSUFFICIENT INFORMATION
- If the JSON is missing the exact data necessary to fully and directly answer the user's question (even if related data exists), treat this as insufficient.
  - Set `"answer": false`
  - In `"reason"` explain why the question cannot be answered from the record (for example: "The employee record does not contain the user's asset id" or "No lock-state or lock-reason is present in the record"). Use plain natural language; do not show JSON keys or raw JSON.

4) EMPLOYEE ID NOT FOUND / NO RECORD
- If the requested employee id is not present in your USER_DATABASE:
  - Set `"answer": false`
  - Put a brief reason such as "No employee record found for the provided id" (or similar natural-language text).

DECISION STRICTNESS
- Return `true` only when the JSON contains the exact information needed to fully answer an issue-type question.
- For general factual queries, return `null` only when the JSON contains the value and you can present it directly.
- If any uncertainty or partial match exists, return `false`.

FORBIDDEN ACTIONS (do not do any of these)
- Do NOT provide anything outside the required JSON structure.
- Do NOT include raw JSON, field/key names, or structured dumps of the record in the "reason".
- Do NOT add additional keys, lists, or arrays.
- Do NOT attempt to answer follow-up questions or perform extra tasks — your job ends after emitting the required JSON.

SAMPLE OUTPUTS (exact JSON only — for reference)
- General factual question (present): 
  {"answer": null, "reason": "Your registered email is xxx@zzz.com."}
- Issue question (present): 
  {"answer": true, "reason": "Your Ultimatix account is locked because of wrong password attempts more than 2 times; last login was 12 hours ago."}
- Insufficient info:
  {"answer": false, "reason": "The employee record does not contain the data needed to answer this question."}
- Employee id not found:
  {"answer": false, "reason": "No employee record found for the provided id."}

END OF INSTRUCTIONS.
""",
    checkpointer=InMemorySaver(),
    state_schema=CustomState,
    response_format=ProviderStrategy(replytype),
)


config_1 = {"configurable": {"thread_id": "11"}}
result = lookup_agent.invoke(
    {"messages": "hii", "employee_id": "2871513"}, config_1
)
print(result["structured_response"])
