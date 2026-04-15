#!/usr/bin/env python3
# agent0.py - AI Agent with memory and tool feedback
# Run: python agent0.py

import subprocess
import os
import asyncio
import aiohttp
import re

# ─── Configuration ───

WORKSPACE = os.path.expanduser("~/.agent0")
MODEL = "minimax-m2.5:cloud"
MAX_TURNS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APPROVED_PATHS = set()

# ─── Memory ───

conversation_history = []
key_info = []

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

# ─── Memory Management ───

def build_context():
    context_parts = []
    if key_info:
        items_xml = "\n".join(f"  <item>{k}</item>" for k in key_info)
        context_parts.append(f"<memory>\n{items_xml}\n</memory>")
    if conversation_history:
        context_parts.append("<history>\n" + "\n".join(conversation_history[-MAX_TURNS*2:]) + "\n</history>")
    return "\n\n".join(context_parts)

def update_memory(user_input, assistant_response, tool_result=None):
    conversation_history.append(f"  <user>{user_input}</user>")
    conversation_history.append(f"  <assistant>{assistant_response}</assistant>")
    if tool_result:
        conversation_history.append(f"  <tool>{tool_result[:500]}</tool>")
    
    while len(conversation_history) > MAX_TURNS * 4:
        conversation_history.pop(0)

async def extract_key_info(user_input, assistant_response):
    extract_prompt = f"""根據這段對話，有沒有需要長期記憶的關鍵資訊？
如果有，用以下格式輸出（最多 2 項）。如果沒有，輸出 <memory></memory>。

<memory>
  <item>要記憶的資訊 1</item>
  <item>要記憶的資訊 2</item>
</memory>

對話：
<user>{user_input}</user>
<assistant>{assistant_response}</assistant>"""
    
    try:
        result = await call_ollama(extract_prompt, "")
        matches = re.findall(r'<item>(.*?)</item>', result, re.DOTALL)
        for item in matches:
            item = item.strip()
            if item and item not in key_info:
                key_info.append(item)
    except:
        pass

# ─── Security Control ───

def is_path_within_allowed(cmd: str) -> tuple[bool, list[str]]:
    """Check if command accesses paths outside allowed directory."""
    external_paths = []
    
    path_patterns = re.findall(r'(?:^|[;\s])([A-Za-z]:[^\s;"\']+|[^\s;"\']+(?:/[^\s;"\']*)*|/[^\s]*?)(?=\s|;|$)', cmd)
    
    for path in path_patterns:
        path = path.strip().strip('"\'')
        if not path:
            continue
        
        if path.startswith('/') and not path.startswith(SCRIPT_DIR):
            if os.path.exists(path) or os.path.dirname(path):
                external_paths.append(path)
                continue
        
        try:
            abs_path = os.path.abspath(path)
            if not abs_path.startswith(SCRIPT_DIR) and os.path.exists(abs_path):
                external_paths.append(abs_path)
        except:
            pass
    
    return len(external_paths) == 0, external_paths

def check_and_approve(cmd: str) -> bool:
    """Check if command accesses external files. Ask for approval if needed."""
    is_safe, external_paths = is_path_within_allowed(cmd)
    
    if is_safe:
        return True
    
    print(f"\n[SECURITY] Command attempts to access files outside program folder:")
    for p in external_paths:
        print(f"  - {p}")
    
    while True:
        response = input("\nApprove this access? (yes/no/always): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        elif response == 'always':
            for p in external_paths:
                APPROVED_PATHS.add(os.path.dirname(p) if os.path.isfile(p) else p)
            return True
        else:
            print("Please enter 'yes', 'no', or 'always'.")

# ─── Agent ───

SYSTEM_PROMPT = """你是 Jarvis，一個有用的 AI 助理。

重要規則：
1. 當你需要執行 shell 命令時，必須用 <shell> 標籤包住命令
2. <shell> 標籤內可以是多行命令（用反斜槓 \\ 或 && 連接）
3. 當你完成所有操作後，用 <end/> 結束你的回覆

流程：
- 如果需要執行命令，輸出 <shell>...</shell>
- 執行完後我會顯示結果
- 如果還需要更多命令，繼續輸出 <shell>
- 當完成所有操作後，輸出 <end/> 表示結束"""

def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    
    print(f"Agent0 - {MODEL}（含記憶功能）")
    print(f"工作區：{WORKSPACE}")
    print(f"安全範圍：{SCRIPT_DIR}")
    print("指令：/quit、/memory（顯示關鍵資訊）\n")
    
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("再見！")
            break
        if user_input.lower() == "/memory":
            print(f"關鍵資訊：{key_info}")
            continue
        
        context = build_context()
        full_prompt = f"{context}\n\n<user>{user_input}</user>" if context else f"<user>{user_input}</user>"
        
        response = asyncio.run(call_ollama(full_prompt, SYSTEM_PROMPT))
        
        tool_result = None
        current_response = response
        
        while True:
            if "<end/>" in current_response:
                response = current_response.split("<end/>")[0].strip()
                break
            
            shell_matches = re.findall(r'<shell>(.+?)</shell>', current_response, re.DOTALL)
            if not shell_matches:
                response = current_response
                break
            
            all_outputs = []
            for cmd in shell_matches:
                cmd = cmd.strip()
                
                if not check_and_approve(cmd):
                    denied_msg = "[ACCESS DENIED] Command blocked by security policy."
                    print(f"\n=== 執行命令 ===\n{cmd}\n\n結果：{denied_msg}\n")
                    all_outputs.append(f"$ {cmd}\n{denied_msg}")
                    continue
                
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
                    output = result.stdout + result.stderr
                    print(f"\n=== 執行命令 ===\n{cmd}\n\n結果：{output if output else '（無輸出）'}\n")
                    all_outputs.append(f"$ {cmd}\n{output if output else '（無輸出）'}")
                except Exception as e:
                    print(f"錯誤：{e}")
                    all_outputs.append(f"$ {cmd}\n錯誤：{e}")
            
            tool_result = (tool_result or "") + "\n" + "\n".join(all_outputs)
            
            follow_up_prompt = f"""<context>{context}</context>

<user>{user_input}</user>
<assistant>{current_response}</assistant>
<output>
{chr(10).join(all_outputs)}
</output>

如果需要更多命令就輸出 <shell>。否則，輸出 <end/> 表示結束："""
            current_response = asyncio.run(call_ollama(follow_up_prompt, SYSTEM_PROMPT))
        
        print(f"\n🤖 {response}\n")
        
        update_memory(user_input, response, tool_result)
        if tool_result:
            asyncio.run(extract_key_info(user_input, response))

if __name__ == "__main__":
    main()
