#!/usr/bin/env python3
# agent0.py - AI Agent with memory and tool feedback
# Run: python agent0.py

import subprocess
import json
import os
import asyncio
import aiohttp
import re

# ─── Security: File Access Guard ───

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def validate_path(requested_path: str) -> bool:
    """Validate if path is within base directory. Returns True if access allowed."""
    abs_requested = os.path.abspath(requested_path)
    try:
        common = os.path.commonpath([SCRIPT_DIR, abs_requested])
        is_internal = common == SCRIPT_DIR or abs_requested.startswith(SCRIPT_DIR + os.sep)
    except ValueError:
        is_internal = False
    
    if is_internal:
        return True
    
    response = input(f"Warning: Accessing external file {requested_path}. Allow? (y/n): ").strip().lower()
    return response == 'y'

def check_command_security(command: str) -> bool:
    """Check if command attempts to access files outside base directory."""
    if not command:
        return True
    
    dangerous_patterns = ['/etc/', '~/', '$HOME', '/root/', '/home/']
    for pattern in dangerous_patterns:
        if pattern in command:
            return False
    
    parts = command.split()
    for part in parts:
        if part.startswith('/') and not part.startswith(SCRIPT_DIR):
            if not validate_path(part):
                return False
    
    return True

# ─── Configuration ───

WORKSPACE = os.path.expanduser("~/.agent0")
MODEL = "minimax-m2.5:cloud"
MAX_TURNS = 5

# ─── Memory ───

conversation_history = []
key_info = []

# ─── Tools ───

TOOLS = [
    {
        "name": "run_command",
        "description": "Run a shell command (use 'cat file' to read, 'echo content > file' or 'printf ... > file' to write)",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"}
            },
            "required": ["command"]
        }
    },
]

# ─── Ollama API ───

async def call_ollama(prompt: str, system: str = "") -> str:
    """Call Ollama API"""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            result = await resp.json()
            return result.get("response", "").strip()

# ─── Tool Execution ───

def execute_tool(name, tool_input):
    print(f"\n=== TOOL EXECUTE ===")
    print(f"Tool: {name}")
    print(f"Input: {tool_input}")
    
    if name == "run_command":
        command = tool_input.get("command", "")
        
        if not check_command_security(command):
            print(f"Result: BLOCKED - Command accesses paths outside allowed directory")
            print(f"=== END ===\n")
            return "Access denied: command attempts to access external paths"
        
        try:
            result = subprocess.run(
                command, shell=True, 
                capture_output=True, text=True, timeout=30,
                cwd=SCRIPT_DIR
            )
            output = result.stdout + result.stderr
            print(f"Result: {output}")
            print(f"=== END ===\n")
            return output if output else "(no output)"
        except Exception as e:
            print(f"Error: {e}")
            print(f"=== END ===\n")
            return f"Error: {e}"
    
    print(f"=== END ===\n")
    return f"Unknown tool: {name}"

# ─── Memory Management ───

def build_context():
    context_parts = []
    if key_info:
        context_parts.append("Key information:\n" + "\n".join(f"- {k}" for k in key_info))
    if conversation_history:
        context_parts.append("Recent conversation:\n" + "\n".join(conversation_history[-MAX_TURNS*2:]))
    return "\n\n".join(context_parts)

def update_memory(user_input, assistant_response, tool_result=None):
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Assistant: {assistant_response}")
    if tool_result:
        conversation_history.append(f"Tool result: {tool_result[:500]}")
    
    while len(conversation_history) > MAX_TURNS * 4:
        conversation_history.pop(0)

async def extract_key_info(user_input, assistant_response):
    extract_prompt = f"""Based on this conversation, should any key information be remembered long-term?
If yes, output a JSON list of key points (max 2). If no, output an empty list [].

Conversation:
User: {user_input}
Assistant: {assistant_response}

Key information to remember:"""
    
    try:
        result = await call_ollama(extract_prompt, "")
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            items = json.loads(match.group(0))
            for item in items:
                if item not in key_info:
                    key_info.append(item)
    except:
        pass

# ─── Agent ───

SYSTEM_PROMPT = """You are Jarvis, a helpful AI assistant.
You have tools - use them to help the user.

Available tools:
- run_command: Run a shell command. Use 'cat file' to read files, 'echo content > file' or 'printf' to write files.

When you need to use a tool, output in this format:
<tool>
{"name": "tool_name", "input": {"key": "value"}}
</tool>

Otherwise, just respond directly."""

def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    
    print(f"Agent0 - {MODEL} (with memory)")
    print(f"Workspace: {WORKSPACE}")
    print("Commands: /quit, /memory (show key info)\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("Goodbye!")
            break
        if user_input.lower() == "/memory":
            print(f"Key info: {key_info}")
            continue
        
        context = build_context()
        full_prompt = f"{context}\n\nUser: {user_input}" if context else f"User: {user_input}"
        
        response = asyncio.run(call_ollama(full_prompt, SYSTEM_PROMPT))
        
        tool_result = None
        current_response = response
        
        while True:
            tool_matches = re.findall(r'<tool>(.+?)</tool>', current_response, re.DOTALL)
            if not tool_matches:
                break
            
            all_tool_outputs = []
            for tool_match in tool_matches:
                try:
                    tool_json = tool_match.strip()
                    tool_data = json.loads(tool_json)
                    tool_name = tool_data.get("name")
                    tool_input = tool_data.get("input", {})
                    
                    tool_output = execute_tool(tool_name, tool_input)
                    all_tool_outputs.append(f"[{tool_name}]: {tool_output}")
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    print(f"Raw tool data: {tool_match}")
                except Exception as e:
                    print(f"Tool error: {e}")
            
            tool_result = (tool_result or "") + "\n" + "\n".join(all_tool_outputs)
            
            follow_up_prompt = f"""Previous context: {context}

User: {user_input}
Previous assistant responses: {current_response}
Tool outputs:
{chr(10).join(all_tool_outputs)}

If you need more tools, output them. Otherwise, provide your final response to the user:"""
            current_response = asyncio.run(call_ollama(follow_up_prompt, SYSTEM_PROMPT))
        
        response = current_response
        
        print(f"\n🤖 {response}\n")
        
        update_memory(user_input, response, tool_result)
        if tool_result:
            asyncio.run(extract_key_info(user_input, response))

if __name__ == "__main__":
    main()
