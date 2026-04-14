from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from context import prompt

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Bedrock client - see Q42 on https://edwarddonner.com/faq if the Region gives you problems
bedrock_client = boto3.client(
    service_name="bedrock-runtime", 
    region_name=os.getenv("DEFAULT_AWS_REGION", "us-east-1")
)

# Bedrock model selection - see Q42 on https://edwarddonner.com/faq for more
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.amazon.nova-2-lite-v1:0")

# Memory storage configuration
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")

# S3 prefixes
S3_PREFIX_CONVERSATIONS = "conversations/"
S3_PREFIX_CONTACTS      = "contacts/"
S3_PREFIX_UNANSWERED    = "unanswered/"

#Tools
TOOLS = [
    {
        "toolSpec": {
            "name": "save_contact",
            "description": (
                "Save a visitor's contact information when they share their email address "
                "in the conversation. Call this automatically as soon as the user provides "
                "their email and do not ask for confirmation."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "The visitor's email address"
                        },
                        "name": {
                            "type": "string",
                            "description": "The visitor's name if mentioned in the conversation"
                        }
                    },
                    "required": ["email"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "save_unanswered_question",
            "description": (
                "Save a question to S3 when you cannot answer based on the context provided. "
                "Do NOT make up answers. Save the question instead so a real person can follow up."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question that could not be answered"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason why the question could not be answered"
                        }
                    },
                    "required": ["question"]
                }
            }
        }
    }
]


# Initialize S3 client if needed
if USE_S3:
    s3_client = boto3.client("s3")


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class Message(BaseModel):
    role: str
    content: str
    timestamp: str


# Memory management functions
def get_memory_path(session_id: str) -> str:
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    """Load conversation history from storage"""
    if USE_S3:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX_CONVERSATIONS}{get_memory_path(session_id)}")
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        # Local file storage
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []


def save_conversation(session_id: str, messages: List[Dict]):
    """Save conversation history to storage"""
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f"{S3_PREFIX_CONVERSATIONS}{get_memory_path(session_id)}",

            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        # Local file storage
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=2)

def execute_save_contact(inputs: Dict, session_id: str) -> str:
    email = inputs.get("email", "").strip()
    name  = inputs.get("name", "").strip()

    if not email or "@" not in email:
        return "Invalid email address — contact not saved."

    contact = {
        "email":      email,
        "name":       name,
        "session_id": session_id,
        "saved_at":   datetime.now().isoformat(),
    }

    key = f"{S3_PREFIX_CONTACTS}{email.replace('@', '_at_').replace('.', '_')}.json"

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(contact, indent=2),
            ContentType="application/json",
        )
        print(f"[Tool] Contact saved: {email}")
        return f"Contact saved successfully for {email}."
    except ClientError as e:
        print(f"[Tool] Failed to save contact: {e}")
        return "Failed to save contact due to a storage error."


def execute_save_unanswered(inputs: Dict, session_id: str) -> str:
    question = inputs.get("question", "").strip()
    reason   = inputs.get("reason", "No reason provided").strip()

    if not question:
        return "No question provided — nothing saved."

    record = {
        "question":   question,
        "reason":     reason,
        "session_id": session_id,
        "timestamp":  datetime.now().isoformat(),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"{S3_PREFIX_UNANSWERED}{timestamp}_{session_id[:8]}.json"

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(record, indent=2),
            ContentType="application/json",
        )
        print(f"[Tool] Unanswered question saved for session {session_id}: {question}")
        return "Question saved. A real person will follow up."
    except ClientError as e:
        print(f"[Tool] Failed to save unanswered question: {e}")
        return "Failed to save the question due to a storage error."


def dispatch_tool(tool_name: str, tool_input: Dict, session_id: str) -> str:
    """Route tool call from Bedrock to the correct function."""
    if tool_name == "save_contact":
        return execute_save_contact(tool_input, session_id)
    elif tool_name == "save_unanswered_question":
        return execute_save_unanswered(tool_input, session_id)
    else:
        return f"Unknown tool: {tool_name}"
    

def call_bedrock(conversation: List[Dict], user_message: str, session_id: str) -> str:
    """Call AWS Bedrock with conversation history and tool support. Runs agentic loop"""

    messages = []

    # Add conversation history (limit to last 25 exchanges)
    for msg in conversation[-50:]:
        messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })

    # Add current user message
    messages.append({
        "role": "user",
        "content": [{"text": user_message}]
    })

    try:
        while True:
            response = bedrock_client.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=messages,
                system=[{"text": prompt()}],
                toolConfig={"tools": TOOLS},
                inferenceConfig={
                    "maxTokens": 2000,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            )

            stop_reason      = response["stopReason"]
            response_message = response["output"]["message"]

            messages.append(response_message)

            if stop_reason == "end_turn":
                for block in response_message["content"]:
                    if "text" in block:
                        return block["text"]
                return "I'm sorry, I couldn't generate a response."

            elif stop_reason == "tool_use":
                tool_results = []

                for block in response_message["content"]:
                   
                    tool_data = block.get("toolUse")
                    if not tool_data:
                        continue

                    tool_name   = tool_data.get("name")
                    tool_input  = tool_data.get("input", {})
                    tool_use_id = tool_data.get("toolUseId")

                    print(f"[Bedrock] Tool called: {tool_name} | input: {tool_input}")

                    result = dispatch_tool(tool_name, tool_input, session_id)

                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content":   [{"text": result}],
                            "status":    "success"
                        }
                    })

                # Only append if we have results — empty content causes ValidationException
                if tool_results:
                    messages.append({
                        "role":    "user",
                        "content": tool_results
                    })
                else:
                    print(f"[Bedrock] stop_reason=tool_use but no tool blocks found in content: {response_message['content']}")
                    break

            else:
                print(f"[Bedrock] Unexpected stopReason: {stop_reason}")
                break

        return "I'm sorry, something went wrong generating a response."

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            print(f"Bedrock validation error: {e}")
            raise HTTPException(status_code=400, detail="Invalid message format for Bedrock")
        elif error_code == 'AccessDeniedException':
            print(f"Bedrock access denied: {e}")
            raise HTTPException(status_code=403, detail="Access denied to Bedrock model")
        else:
            print(f"Bedrock error: {e}")
            raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "AI Digital Twin API (Powered by AWS Bedrock)",
        "memory_enabled": True,
        "storage": "S3" if USE_S3 else "local",
        "ai_model": BEDROCK_MODEL_ID
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "use_s3": USE_S3,
        "bedrock_model": BEDROCK_MODEL_ID
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Load conversation history
        conversation = load_conversation(session_id)

        # Call Bedrock for response
        assistant_response = call_bedrock(conversation, request.message, session_id)

        # Update conversation history
        conversation.append(
            {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save conversation
        save_conversation(session_id, conversation)

        return ChatResponse(response=assistant_response, session_id=session_id)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history"""
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/unanswered")
async def list_unanswered():
    """List all unanswered questions saved in S3 for human review."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX_UNANSWERED)
        items = []
        for obj in response.get("Contents", []):
            file = s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
            items.append(json.loads(file["Body"].read().decode("utf-8")))
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return {"count": len(items), "questions": items}
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contacts")
async def list_contacts():
    """List all saved contacts from S3."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX_CONTACTS)
        contacts = []
        for obj in response.get("Contents", []):
            file = s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
            contacts.append(json.loads(file["Body"].read().decode("utf-8")))
        contacts.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return {"count": len(contacts), "contacts": contacts}
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)