"""
Optimized TCS onboarding chatbot workflow
- Single clean module: environment, tools, agents, workflow, and run loop
- Preserves original logic: lookup_agent -> sop_agent -> llmjudge_agent
- Improvements: no duplicate imports, clear helper functions, logging, error handling,
  consistent type hints, and minimal comments.
"""

import os
import logging
from typing import Annotated, Any, Dict, Optional, TypedDict
from dotenv import load_dotenv
from pprint import pprint

# LangChain / LangGraph imports (assume your installed versions match these APIs)
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent, AgentState
from langchain.agents.structured_output import ProviderStrategy
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.agents.middleware import HumanInTheLoopMiddleware, dynamic_prompt, ModelRequest
from pydantic import Field
from typing_extensions import TypedDict as TE_TypedDict

# -------------------------
# Configuration & Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("tcs_onboard_bot")
load_dotenv()

# Single environment init
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

if AZURE_KEY:
    os.environ["AZURE_OPENAI_API_KEY"] = AZURE_KEY
if AZURE_ENDPOINT:
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT
if AZURE_DEPLOY:
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = AZURE_DEPLOY
if OPENAI_API_VERSION:
    os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION

# Initialize LLM
llm = init_chat_model("azure_openai:gpt-4.1", azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"))

# Embedding model & retriever defaults
EMBED = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
TOP_K = 10
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 50

# -------------------------
# Utility: create file-based retriever tool
# -------------------------
def make_retriever_tool_from_pdf(pdf_path: str, tool_name: str, description: str):
    """
    Returns a tool function that does k-similarity retrieval from the given PDF.
    The returned object is a langchain @tool-decorated callable with .name and .description.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=PDF_CHUNK_SIZE, chunk_overlap=PDF_CHUNK_OVERLAP).split_documents(docs)
    vs = FAISS.from_documents(chunks, EMBED)

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    @tool
    def _tool(query: str) -> str:
        log.info("Retriever tool '%s' invoked", tool_name)
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)

    _tool.name = tool_name
    _tool.description = description
    return _tool

# Example: create your onboarding retriever tool
ONBOARDING_PDF = "TCS_Intern_Onboarding_Instructions.pdf"
tcs_onboarding_tool = make_retriever_tool_from_pdf(
    ONBOARDING_PDF,
    "tcs_onboarding_agent",
    "Answers onboarding questions strictly from the TCS Intern Onboarding PDF."
)

# -------------------------
# In-memory "user database" (kept same as your original)
# -------------------------
USER_DATABASE: Dict[str, Dict[str, Any]] = {
    "2871513": {
        "name": "Arun",
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
        "name": "Priya",
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
        "name": "Rahul",
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
        "name": "Sneha",
        "employee_id": "2890077",
        "ultimatix_login_last": "5 hours ago",
        "domain_account_locked": {"state": True, "reason": "Multiple wrong login attempts"},
        "ultimatix_account_locked": {"state": True, "reason": "Authenticator not registered"},
        "domain_region": "USA",
        "asset_id": None,
        "email": "sneha.p@tcs.com",
        "bgv_status": "Initiated",
        "project": "Onboarding",
        "grade": "Y",
    },
}

# -------------------------
# Tools that perform side-effects (safe, small stubs)
# -------------------------
@tool
def mail_tool() -> str:
    return "Your mail has been sent."

@tool
def otp_tool() -> str:
    return "OTP has been sent."

@tool
def servicenow_tool() -> str:
    return "Your ServiceNow ticket has been submitted."

@tool
def live_agent_tool() -> str:
    return "A live agent has been connected."

# -------------------------
# Tool that accesses USER_DATABASE (uses AgentState via ToolRuntime)
# -------------------------
class CustomState(AgentState):
    employee_id: str

@tool
def get_employee_details(runtime: ToolRuntime[None, CustomState]) -> str:
    """
    Fetch and return a trimmed employee profile dictionary (stringified)
    or "Employee not found in system."
    NOTE: calling agent code should parse this if needed.
    """
    employee_id = runtime.state.get("employee_id")
    log.info("get_employee_details called for id=%s", employee_id)
    if not employee_id or employee_id not in USER_DATABASE:
        return "Employee not found in system."
    return str(USER_DATABASE[employee_id])  # returning a readable representation

# -------------------------
# Response Schemas (TypedDicts) used by structured-output agents
# -------------------------
class ContactInfo(TE_TypedDict):
    answer: str
    call_llm_judge: bool
    emotion: str

class ReplyType(TE_TypedDict):
    answer: bool
    reason: str

# -------------------------
# Create Agents
# -------------------------
# SOP agent: helps with onboarding issues, uses tcs_onboarding_tool
sop_agent = create_agent(
    llm,
    tools=[tcs_onboarding_tool],
    system_prompt="""You are the TCS Onboarding Assistant helping new employees and interns (mostly freshers).
- Use the tool to retrieve exact onboarding doc content when relevant.
- Be empathetic, short (3-4 sentences or small bullets), and beginner friendly.
- Start with empathy, provide 1-2 immediate checks, then ask one follow-up question.
- Track troubleshooting attempts; escalate after 3 failed attempts by returning call_llm_judge = true.
- Return structured output matching ContactInfo with keys: answer, call_llm_judge, emotion.
""",
    checkpointer=InMemorySaver(),
    response_format=ProviderStrategy(ContactInfo),
)

# Lookup agent: determines whether question can be answered from employee record
lookup_agent = create_agent(
    llm,
    tools=[get_employee_details],
    system_prompt="""You are a Knowledge-Base Lookup Agent.
- Use get_employee_details tool whenever an employee id is available.
- Return EXACTLY a JSON with two keys: 'answer' and 'reason' following the strict rules:
  * For general factual queries present in the record -> answer: null, reason: <human text>
  * For issue/troubleshooting queries present in the record -> answer: true, reason: <human text>
  * For insufficient info or id not found -> answer: false, reason: <why>
- No extra prose.
""",
    checkpointer=InMemorySaver(),
    response_format=ProviderStrategy(ReplyType),
    state_schema=CustomState,
)

# Judge agent: dynamic prompt + human-in-the-loop middleware to approve tool calls
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    bot_type = request.runtime.context.get("bot_type", "")
    issue = request.runtime.context.get("issue", "")
    base_prompt = f"""
You are an LLM Judge Support Agent.
- Detect emotion and start with an empathetic sentence.
- Based on bot_type={bot_type} and issue summary='{issue}', suggest the next best tool to call.
- Keep replies human, short, and include practical next steps.
"""
    return base_prompt

llmjudge_agent = create_agent(
    llm,
    tools=[mail_tool, otp_tool, servicenow_tool, live_agent_tool],
    middleware=[
        user_role_prompt,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "mail_tool": {"allowed_decisions": ["approve", "reject"]},
                "otp_tool": {"allowed_decisions": ["approve", "reject"]},
                "servicenow_tool": {"allowed_decisions": ["approve", "reject"]},
                "live_agent_tool": {"allowed_decisions": ["approve", "reject"]},
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    context_schema={"bot_type": str, "issue": str},
    checkpointer=InMemorySaver(),
)

# -------------------------
# Workflow Graph: nodes and routing
# -------------------------
class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    employee_id: Optional[str]
    lookup_decision: Optional[bool]
    call_llm_judge: Optional[bool]
    bot_type: Optional[str]
    issued_by_agents: Optional[str]
    judge_res: Optional[str]
    emotion: Optional[str]
    __interrupt__: Optional[dict]

def run_lookup_agent(state: WorkflowState):
    # call lookup_agent and return minimal structured outputs
    log.info("Running lookup agent for employee_id=%s", state.get("employee_id"))
    result = lookup_agent.invoke(
        {"messages": state["messages"], "employee_id": state["employee_id"]},
        {"configurable": {"thread_id": "LOOKU"}}
    )
    return {
        "lookup_decision": result["structured_response"]["answer"],
        "issued_by_agents": result["structured_response"]["reason"],
        "bot_type": "lookup_bot",
        "messages": result["messages"],
    }

def run_sop_agent(state: WorkflowState):
    log.info("Running sop agent")
    result = sop_agent.invoke(
        {"messages": state["messages"]},
        {"configurable": {"thread_id": "SO"}}
    )
    return {
        "issued_by_agents": result["structured_response"]["answer"],
        "call_llm_judge": result["structured_response"]["call_llm_judge"],
        "bot_type": "sop_bot",
        "emotion": result["structured_response"]["emotion"],
        "messages": result["messages"],
    }

def run_llm_judge_agent(state: WorkflowState):
    log.info("Running llm judge agent")
    result = llmjudge_agent.invoke(
        {"messages": state["messages"]},
        context={"bot_type": state.get("bot_type", ""), "issue": state.get("issued_by_agents", "")},
        config={"configurable": {"thread_id": "JUDG"}}
    )
    # capture messages and potential interrupt payload
    payload = {"judge_res": result["messages"][-1].content, "messages": result["messages"]}
    # if middleware triggered an interrupt, the graph engine will include it in the returned state
    return payload

def route_after_lookup(state: WorkflowState):
    lookup = state.get("lookup_decision")
    if lookup is None:
        return END
    if lookup is True:
        return "judge"
    return "sop"

def route_after_sop(state: WorkflowState):
    return "judge" if state.get("call_llm_judge") else END

# Build graph
checkpointer = InMemorySaver()
workflow = StateGraph(WorkflowState)
workflow.add_node("lookup", run_lookup_agent)
workflow.add_node("judge", run_llm_judge_agent)
workflow.add_node("sop", run_sop_agent, interrupt_before=["user_input"])
workflow.set_entry_point("lookup")
workflow.add_conditional_edges("lookup", route_after_lookup, {END: END, "judge": "judge", "sop": "sop"})
workflow.add_conditional_edges("sop", route_after_sop, {"judge": "judge", END: END})
workflow.add_edge("judge", END)
workflow.add_edge("sop", END)
graph = workflow.compile(checkpointer=checkpointer)

# -------------------------
# Interactive run-loop
# -------------------------
def run_repl():
    config_1 = {"configurable": {"thread_id": "87"}}
    pending_interrupt = False
    saved_interrupt_payload = None

    print("🤖 Chatbot ready — type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if user_input.lower() == "exit":
            break

        # If resuming from a human-approval interrupt
        if pending_interrupt and saved_interrupt_payload:
            print("\n🔄 Resuming after approval decision...")
            final_state = graph.invoke(None, config=config_1)  # resume
            pending_interrupt = False
            saved_interrupt_payload = None
        else:
            # Build inputs
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "employee_id": "2871513",  # keep fixed employee id as in your original loop
            }

            final_state = None
            # Stream walk through the graph
            for output in graph.stream(inputs, config=config_1):
                # the engine yields intermediate node outputs
                for node_name, node_value in output.items():
                    pprint(f"Node: {node_name}")
                    final_state = node_value

            # If an interrupt is present, the graph engine should return it in final_state
            if final_state and isinstance(final_state, dict) and "__interrupt__" in final_state:
                interrupt_data = final_state["__interrupt__"]
                print("\n🔴 TOOL APPROVAL NEEDED:")
                pprint(interrupt_data[0].value)
                decision = ""
                while decision not in ("y", "n"):
                    decision = input("\nApprove? (y/n): ").strip().lower()
                if decision == "y":
                    # Simulate an approval — in a real environment, you'd pass decision back to middleware
                    print("Approved. Resuming...")
                    pending_interrupt = True
                    saved_interrupt_payload = interrupt_data
                    continue
                else:
                    print("Rejected. No tool executed. You may continue.")
                    # Reset and continue main loop
                    pending_interrupt = False
                    saved_interrupt_payload = None
                    continue

        # Print output
        print("\n✅ Workflow completed!")
        if final_state and "messages" in final_state and final_state["messages"]:
            last_msg = final_state["messages"][-1]
            print(f"🤖 Bot: {last_msg.content}")
        else:
            # Fallback short summary
            print("🤖 Output:", final_state)

if __name__ == "__main__":
    run_repl()
