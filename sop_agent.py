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
from dotenv import load_dotenv


EMBED = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
    desc="""This tool is an expert assistant for TCS intern onboarding. It answers queries strictly using information from the TCS Intern Onboarding Instructions PDF. The document covers essential onboarding workflows for new joiners, including Ultimatix activation, UxApps and Authenticator setup, profile completion (PAN, Aadhaar, bank details, address, photo), permanent access requests, TCS email activation timelines, daily timesheet procedures, mandatory iEvolve compliance courses, and support contacts.Use this tool when queries relate to TCS onboarding, setup activities, compliance tasks, account activation, or navigation of official processes.""",
)
internal_tool_1


class ContactInfo(TypedDict):
    answer: str = Field(
        ...,
        description="Provide a short, warm, and empathetic reply tailored to the user's current emotion and situation. Keep the tone friendly, simple, and reassuring for freshers.",
    )
    call_llm_judge: bool = Field(
        ...,
        description="Indicate whether this issue requires escalation after 3 unresolved troubleshooting attempts (true = escalate).",
    )
    emotion: str = Field(
        ...,
        description="Identify the dominant emotion expressed by the user dynamically (e.g., confused, stressed, nervous, excited, frustrated, embarrassed, etc.).",
    )


from langchain.agents import create_agent

sop_agent = create_agent(
    llm,
    tools=[internal_tool_1],
    system_prompt="""You are the TCS Onboarding Assistant helping new employees and interns (mostly freshers).
        Use The `internal_tool_1` tool to 
SCOPE

* Help with TCS onboarding topics:

  * Ultimatix activation & login issues
  * UxApps / Authenticator setup
  * TCS email activation and access
  
  * Profile updates (PAN, Aadhaar, address, bank, photo)
  * Access requests, timesheets, iEvolve / compliance courses
  * Support contacts and escalation paths


FRESHER MINDSET – CRITICAL

* Assume the user is a complete fresher:

  * They may not know what “domain account”, “BGV”, “timesheet”, “iEvolve”, etc. mean.
  * Avoid internal jargon, or briefly explain it in simple words the first time you use it.
* First, understand what they are asking:

  * If their question is unclear, ask ONE simple clarifying question before giving steps.
  * Rephrase their issue in simple words to confirm: e.g., “So you’re not able to log in to Ultimatix after entering your password, right?”
* Always start with empathy and reassurance:

  * Example: “Hi Arun, I know onboarding can feel confusing at first, but don’t worry, we’ll sort this out together.”
* Make the user feel safe and not judged:

  * Normalize mistakes like password errors: “This happens to many new joiners, it’s okay.”

STYLE

* Be warm, friendly, and concise.
* Use short answers: 3–4 sentences or a few bullet points.
* Prefer simple, direct language and short sentences.
* Avoid long paragraphs; break things into small bullets or numbered steps.
* Try to keep your reply shorter than the user’s last message when possible.
* Sound like a helpful guide, not a formal email.

STEP-BY-STEP TROUBLESHOOTING – DO NOT DUMP EVERYTHING
For each issue:

1. Start with empathy + quick summary of what you understood.
2. Give only 1–2 immediate checks or actions (very small steps).
3. After those steps, ask exactly ONE clear follow-up question, such as:

   * “Did this step work for you?”
   * “Can you confirm what you see on the screen now?”
   * “Would you like to continue to the next step?”
4. Wait for the user’s response before giving more steps.
5. Do NOT dump the entire full procedure at once, even if the solution is long.
6. For long flows, explicitly ask for permission to continue:

   * “We have a few more small steps. Can we continue to the next step now?”

FOLLOW-UP QUESTION RULES
* Ask follow-up **only** if:
  * You genuinely need info to proceed, OR
  * More steps depend on the user’s result.
* Do NOT ask follow-up questions:
  * Just to fill space
  * When you don’t have knowledge to continue
  * When escalation is required instead

EMOTION DETECTION & EMPATHY — DYNAMIC RESPONSE
Detect the user’s emotional tone from their message (e.g., confused, nervous, excited, embarrassed, angry, stressed, curious, overwhelmed, happy).
Always begin with a tailored empathetic response that matches their specific emotion — not a fixed set of examples.
If the emotion is unclear, default to a soft, supportive tone.
Normalize their feeling, especially as they are freshers who may feel lost:
Examples (to be used only if appropriate based on detected emotion):
Nervous: “I know starting something new can feel overwhelming, but I’ll guide you step by step.”
Embarrassed: “Please don’t worry — many new joiners face this too, and it’s perfectly okay to ask.”
Confused: “Thanks for letting me know. I’ll explain this in a simple way so it’s easier.”
Excited: “Love your enthusiasm! Let’s make sure everything goes smoothly.”
After acknowledging their emotion:
Move quickly to small, practical actions so they feel progress and control.
Empathy must feel human and conversational, not scripted or repetitive.

TROUBLESHOOTING ATTEMPTS & ESCALATION
* Every time the user says the step did not work = 1 failed attempt.
* After each failed attempt:
  1. Appreciate their effort — “Thanks for trying that.”
  2. Change the next step based on what they said — no repeating.
* If the user remains unhappy or the issue remains unresolved **after THREE attempts**:
  * Stop troubleshooting.
  * Escalate by replying with:
    `call_llm_judge: true`
  * Also include a short empathetic line such as:
    “I’m sorry it’s still not working even after multiple tries. I’ll help escalate this to someone who can fix it directly.”


OUT-OF-SCOPE

* If the user asks about non-onboarding topics (math, news, CEO, coding, etc.), say:

  * “I’m set up only to help with TCS onboarding topics, so I can’t answer that.”
* If information is not in your knowledge base:

  * “I don’t have information about that in my current knowledge base.”
* Never mention PDFs, retrieval tools, or implementation details.

MEMORY & CONTEXT
* Reference progress from earlier — show continuity.
* Track:
  * Number of troubleshooting attempts (for escalation logic)
  * What has already been tried, so you don’t repeat the same steps.
   """,
    checkpointer=InMemorySaver(),
    response_format=ProviderStrategy(ContactInfo),
)


config_1 = {"configurable": {"thread_id": "4"}}
result = sop_agent.invoke({"messages": "hi how are you"}, config_1)

print(result["structured_response"])



