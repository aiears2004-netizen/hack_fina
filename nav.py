import os
from typing import Annotated, TypedDict, Literal, Dict, Any
from fastapi import FastAPI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from datetime import datetime
from pydantic import BaseModel

# KB Articles - USER ONLY
KB_ARTICLES = {
    "office_access": {
        "content": """**Office Access Process (USER):**
1. **Submit Form**: [Office Access Form](https://forms.company.com/office-access)
2. **Processing**: 24-48 hours
3. **Required**: Employee ID + Manager approval  
4. **Status**: Check email after 24hrs
5. **Help**: facilities@company.com"""
    }
}

# Global notification log - VIP ONLY
NOTIFICATIONS = []
REGIONAL_EMAILS = {
    "Chennai": "chennai-facilities@company.com",
    "Mumbai": "mumbai-facilities@company.com", 
    "Bangalore": "bangalore-facilities@company.com"
}

app = FastAPI(title="USER=KB | VIP=Instant Access ")

class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_type: str
    region: str | None
    access_granted: bool
    employee_id: str

class ChatRequest(BaseModel):
    user_id: str
    user_type: Literal["VIP", "USER"]
    message: str

@tool
def show_process() -> str:
    """Show USER office access process."""
    return KB_ARTICLES["office_access"]["content"]

@tool
def extract_location(message: str) -> str:
    """Extract VIP office location."""
    locations = ["Chennai", "Mumbai", "Bangalore"]
    for loc in locations:
        if loc.lower() in message.lower():
            return f"VIP Location: {loc}"
    return "Bangalore (default)"

@tool
def grant_vip_access(employee_id: str, region: str) -> str:
    """Grant INSTANT VIP access."""
    notification = {
        "timestamp": datetime.now().isoformat(),
        "type": "VIP_ACCESS_GRANTED", 
        "employee_id": employee_id,
        "region": region,
        "team_email": REGIONAL_EMAILS.get(region, "facilities@company.com")
    }
    NOTIFICATIONS.append(notification)
    
    print(f" VIP INSTANT ACCESS: {employee_id} → {region}")
    return f""" **VIP ACCESS GRANTED INSTANTLY!** 🎉
        **Employee**: {employee_id}
        **Office**: {region}
        **Status**: Badge ACTIVATED NOW
        **Team**: {notification['team_email']} notified"""

# TOOLS FILTERED BY USER TYPE
def get_tools(user_type: str):
    if user_type == "VIP":
        return [extract_location, grant_vip_access]  # VIP Instant Access
    else:  # USER
        return [show_process]  # USER KB Process Only

# Azure OpenAI
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

def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """VIP=Instant | USER=Process."""
    user_type = state.get("user_type", "USER")
    access_granted = state.get("access_granted", False)
    
    if user_type == "VIP":
        system_prompt = f"""VIP Office Access Assistant - INSTANT PROCESSING.

You handle VIPs ONLY. Grant access IMMEDIATELY for urgent requests.

RULES:
- ANY urgent/ASAP/emergency → extract_location + grant_vip_access NOW
- Skip KB articles - VIPs don't need process
- Be fast and direct

Status: Access granted = {access_granted}
Available: extract_location, grant_vip_access"""
        
        tools = [extract_location, grant_vip_access]
    else:  # USER
        system_prompt = f"""USER Office Access Assistant - STANDARD PROCESS.

Users follow normal KB process. NO instant access.

RULES:  
- Answer questions directly
- Show process ONLY if asked "how", "process", "form", "steps"
- Guide to form politely
- NO grant_vip_access (USER only)
- Be helpful and patient

Status: Access granted = {access_granted}
Available: show_process"""
        
        tools = [show_process]
    
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages", n=6),
    ])
    
    response = prompt | llm_with_tools
    ai_message = response.invoke(state)
    
    return {"messages": [ai_message]}

def create_graph():
    workflow = StateGraph(AgentState)
    
    # Dynamic tool node per user_type (simplified - uses USER tools by default)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode([show_process]))  # Base tools
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", 
        tools_condition,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer), checkpointer

graph, checkpointer = create_graph()

@app.post("/chat")
async def chat(request: ChatRequest):
    """USER=KB Process | VIP=Instant Access."""
    thread_id = f"conv-{request.user_id}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Safe state access
    try:
        current_state = checkpointer.get(config)
        access_granted = current_state.get("access_granted", False) if current_state else False
    except:
        access_granted = False
    
    # Complete input state
    input_state: Dict[str, Any] = {
        "messages": [HumanMessage(content=request.message)],
        "user_type": request.user_type,
        "region": None,
        "access_granted": access_granted,
        "employee_id": request.user_id.upper()
    }
    
    result = graph.invoke(input_state, config)
    
    words = request.message.split()
    email_match = next((w for w in words if '@' in w), None)
    employee_id = email_match or request.user_id.upper()
    
    final_msg = result["messages"][-1]
    
    if request.user_type == "VIP" and "ACCESS GRANTED" in final_msg.content:
        print(f" VIP [{request.user_id}] INSTANT ACCESS!")
    else:
        print(f" [{request.user_type}] {request.user_id}: {final_msg.content[:50]}...")
    
    return {
        "response": final_msg.content,
        "user_type": request.user_type,
        "access_granted": result.get("access_granted", False),
        "employee_id": employee_id,
        "conversation": {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "message_count": len(result["messages"])
        },
        "notifications_count": len(NOTIFICATIONS)
    }

@app.get("/conversation/{user_id}")
async def get_history(user_id: str):
    """Get conversation history."""
    thread_id = f"conv-{user_id}"
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = checkpointer.get(config)
        if not state:
            return {"message": "No conversation found"}
        
        messages = state["messages"][-6:]
        return {
            "user_id": user_id,
            "user_type": state.get("user_type", "USER"),
            "access_granted": state.get("access_granted", False),
            "messages": [
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": str(m.content[:100])
                }
                for m in messages
            ]
        }
    except:
        return {"message": "No conversation"}

@app.delete("/conversation/{user_id}")
async def clear_history(user_id: str):
    """Clear conversation."""
    thread_id = f"conv-{user_id}"
    config = {"configurable": {"thread_id": thread_id}}
    checkpointer.clear(config)
    return {"message": f"Cleared for {user_id}"}

@app.get("/notifications")
async def notifications():
    """VIP notifications only."""
    return {"notifications": NOTIFICATIONS[-3:], "total": len(NOTIFICATIONS)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 USER=KB Process | VIP=Instant Access ")
    uvicorn.run(app, host="0.0.0.0", port=8000)



# {
#     "user_id": "Navin",
#     "user_type": "VIP",  "USER"
#     "message": "It is for Bangalore location"
# }