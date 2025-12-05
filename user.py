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

top_k = 10
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

EMBED = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def make_retriever_tool_from_text(file, name, desc):
    docs = PyPDFLoader(file).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    vs = FAISS.from_documents(chunks, EMBED)
    retriever = vs.as_retriever(
        search_type="similarity",  # or "mmr" for maximal marginal relevance
        search_kwargs={"k": top_k},
    )

    @tool  # Use @tool decorator without arguments
    def tool_func(query: str) -> str:
        """Retrieve documents based on query."""
        print(f"📚 Using tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)

    # Set name and description after decoration
    tool_func.name = name
    tool_func.description = desc

    return tool_func


internal_tool_1 = make_retriever_tool_from_text(
    file="TCS_Intern_Onboarding_Instructions.pdf",
    name="tcs_onboarding_agent",
    desc="This agent specializes in answering queries based solely on the 'TCS Intern Onboarding Instructions' PDF. It contains detailed guidance for new joiners at TCS, including Ultimatix activation, UxApps and Authenticator setup, mandatory profile updates (PAN, Aadhaar, address, bank details, photo), permanent access request creation, TCS email activation timelines, daily timesheet submission steps, mandatory iEvolve compliance courses, support contacts, and general onboarding workflows. The agent provides accurate, concise, step-by-step answers strictly from this PDF and rejects questions outside the document’s scope. Use this agent whenever a query is related to TCS onboarding, initial setup, compliance tasks, or process navigation.",
)

USER_DATABASE = {
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


# Define custom state with employee_id
class CustomState(AgentState):
    employee_id: str


@tool
def get_employee_details(runtime: ToolRuntime[None, CustomState]) -> str:
    """Get employee's account details, account status, and onboarding progress. Use this to provide personalized help based on the employee's actual situation."""
    employee_id = runtime.state.get("employee_id")

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

    details = f"""
EMPLOYEE: {employee['name']} (ID: {employee_id})
Email: {employee['email']}
Region: {employee['domain_region']}
Project: {employee['project']}
Grade: {employee['grade']}
BGV Status: {employee['bgv_status']}
Last Ultimatix Login: {employee['ultimatix_login_last']}
Asset ID: {employee['asset_id'] or 'Not assigned'}

ACCOUNT STATUS:
{chr(10).join(account_status)}
"""

    return details


tools = [internal_tool_1, get_employee_details]


class ContactInfo(TypedDict):

    #   "answer": "<write an empathetic, supportive, warm answer to the user>",   "confidence": <confidence score between 0 and 100>,   "emotion": "<emotion you infer from the user's message>"
    answer: str = Field(
        ...,
        description="write an empathetic, supportive, warm answer to the user",
    )
    confidence: str = Field(..., description="confidence score between 0 and 100")
    emotion: Literal[
        "Joy", "Anger", "Sadness", "Fear", "Anxiety", "Disgust", "Surprise"
    ] = Field(..., description="emotion you infer from the user's message{user_input}")


agent = create_agent(
    llm,
    tools,
    system_prompt="""You are the TCS Onboarding Assistant helping new employees and interns.

SCOPE
- Help with TCS onboarding topics:
  - Ultimatix activation & login issues
  - UxApps / Authenticator setup
  - TCS email activation and access
  - Profile updates (PAN, Aadhaar, address, bank, photo)
  - Access requests, timesheets, iEvolve / compliance courses
  - Support contacts and escalation paths

PERSONALIZATION - VERY IMPORTANT
- ALWAYS use get_employee_details tool FIRST to fetch the employee's current account status
- Use the employee's name and actual account status in your response
- If they have account locks or issues, address them specifically:
  * If Ultimatix is locked: explain WHY (e.g., "password attempts") and what to do
  * If domain account is locked: provide specific unlock steps
  * If BGV is Pending/Initiated: mention this affects certain access
- Tailor your troubleshooting to their actual situation, not generic steps

STYLE
- Keep responses short and clear (4–6 sentences max or bullet points)
- Be warm but concise - no long paragraphs
- Use simple language for new joiners
- When troubleshooting, ask ONE clear follow-up question and wait for response

STEP-BY-STEP TROUBLESHOOTING
- Don't dump entire procedures at once
- For each issue:
  1. Give 1–2 immediate checks or actions
  2. Ask exactly one clear follow-up question
  3. Wait before giving next steps
- Example: "Hi Arun, I see your Ultimatix is locked due to password attempts. Did you try the 'Forgot Password' option on the login screen?"

OUT-OF-SCOPE
- If user asks about non-onboarding topics (math, news, CEO, etc.):
  "I'm set up only to help with TCS onboarding topics, so I can't answer that."
- If information not in knowledge base:
  "I don't have information about that in my current knowledge base."
- Never mention PDFs, retrieval tools, or implementation details

MEMORY
- Remember and use employee name naturally in responses
- Reference their specific issues by name: "Hi Priya, I see your domain account has an expired password..."
- Don't repeat ID unless needed for escalation

EMOTION DETECTION
- Detect user's emotion from their message (frustrated, anxious, confused, excited, etc.)
- Respond with appropriate empathy and reassurance
- Address frustration with solutions, not explanations
""",
    checkpointer=InMemorySaver(),
    state_schema=CustomState,
    response_format=ProviderStrategy(ContactInfo),
)


from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
import uuid

# Initialize
# llm = init_chat_model("gpt-4o-mini")
# checkpointer = InMemorySaver()

# Create agent
# agent = create_agent(
#     llm,
#     tools,
#     system_prompt="You are the TCS Onboarding Instructions Agent...",
#     checkpointer=checkpointer
# )


def chat_with_agent(thread_id: str = None):
    """Interactive chatbot loop with persistence."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}
    messages = []

    print("🎓 TCS Onboarding Agent")
    print("Type 'exit' to quit | 'new' for new conversation\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            messages = []
            print("\n🔄 New conversation started\n")
            continue

        if not user_input:
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Invoke agent with config
        try:
            result = agent.invoke(
                {"messages": messages, "employee_id": "2871513"}, config
            )

            # Extract agent response
            assistant_message = result["messages"][-1].content
            print(f"\n🤖 Agent: {assistant_message}\n")

            # Update messages with agent response
            messages.append({"role": "assistant", "content": assistant_message})

        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


# Run the chatbot
if __name__ == "__main__":
    # For single conversation
    chat_with_agent(thread_id="user_session_1")

    # OR for multiple independent sessions:
    # chat_with_agent()  # Creates unique thread_id
