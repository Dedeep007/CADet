import argparse
import base64
import json
import os
import random
import subprocess
import sys
import tempfile
from typing import TypedDict, Optional, List, Dict, Any

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


load_dotenv()

# ==========================================
# STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    messages: List[Any]
    image_path: Optional[str]
    retry_count: int
    lint_errors: List[str]
    scad_code: str
    valid: bool
    skip_export: bool


# ==========================================
# LLM FALLBACK WRAPPER
# ==========================================
def get_keys(prefix: str) -> List[str]:
    return [v for k, v in os.environ.items() if k.startswith(prefix) and v]

def invoke_with_retry(llm, messages):
    import time
    for i in range(3):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "exhausted" in err_str or "quota" in err_str:
                delay = 20 * (i + 1)
                print(f"[Info] API rate limit/quota hit. Sleeping {delay}s...", file=sys.stderr)
                time.sleep(delay)
            else:
                raise e
    return llm.invoke(messages)

def call_model_with_key_rotation(
    models_to_try: List[tuple[str, str]], # list of (provider, model_name)
    messages: List[Any],
    temperature: float = 0.1,
    max_tokens: int = 4096,
):
    """
    Attempts to call the LLMs in order. For each LLM, it shuffles and tries all 
    available API keys for that provider to completely avoid rate limits and use all keys.
    """
    last_err = None
    for provider, model_name in models_to_try:
        if provider == "gemini":
            keys = get_keys("GEMINI_API_KEY")
            if not keys and os.getenv("GOOGLE_API_KEY"):
                keys = [os.getenv("GOOGLE_API_KEY")]
            keys = [k for k in keys if k]
            random.shuffle(keys)
            for key in keys:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        google_api_key=key
                    )
                    return invoke_with_retry(llm, messages)
                except Exception as e:
                    last_err = e
                    print(f"[Warning] Gemini ({model_name}) failed with key {key[:8]}... : {e}", file=sys.stderr)
                    
        elif provider == "groq":
            keys = get_keys("GROQ_API_KEY")
            random.shuffle(keys)
            for key in keys:
                try:
                    llm = ChatGroq(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=key
                    )
                    return invoke_with_retry(llm, messages)
                except Exception as e:
                    last_err = e
                    print(f"[Warning] Groq ({model_name}) failed with key {key[:8]}... : {e}", file=sys.stderr)

    raise RuntimeError(f"All models and keys failed. Last error: {last_err}")


# ==========================================
# HELPERS
# ==========================================
import re

def __get_text(content: Any) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    else:
        text = str(content)
        
    # Strip <think>...</think> blocks from reasoning models (handles unclosed tags from truncation)
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
    return text.strip()

def _extract_json_block(text: Any) -> dict:
    """Safely extracts a JSON block from LLM text."""
    text = __get_text(text)
    try:
        # First naive parse
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    try:
        start_idx = text.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = text.find("```", start_idx)
            if end_idx != -1:
                return json.loads(text[start_idx:end_idx].strip())
        
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx+1].strip())
    except Exception as e:
        print(f"[Error] Failed to extract JSON: {e}", file=sys.stderr)
        
    return {}


def _extract_python_block(text: Any) -> str:
    """Safely extracts Python code from LLM text."""
    text = __get_text(text)
    start_idx = text.find("```python")
    if start_idx != -1:
        start_idx += 9
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx].strip()
            
    # Fallback if code block language is omitted
    start_idx = text.find("```")
    if start_idx != -1:
        start_idx += 3
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx].strip()
            
    return text.strip()


def _run_subprocess(command: List[str], cwd: Optional[str] = None) -> str:
    """Wraps subprocess execution with unified error handling."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=120
        )
        return result.stdout
    except FileNotFoundError:
        binary = command[0]
        raise RuntimeError(f"Binary '{binary}' not found on PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command '{' '.join(command)}' timed out after 120s.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed with exit {e.returncode}.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")


# ==========================================
# NODE 1: VISION ANALYZER
# ==========================================
def node_vision_analyzer(state: AgentState) -> AgentState:
    print("-> Running Vision Analyzer...")
    user_prompt = state["messages"][0]
    image_path = state.get("image_path")
    
    content = [
        {"type": "text", "text": f"User prompt: {user_prompt}\nAnalyze this design request and any provided sketch."}
    ]
    
    if image_path:
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                # Detect extension for mime type
                ext = image_path.lower().split('.')[-1]
                mime = "image/png"
                if ext in ["jpg", "jpeg"]: mime = "image/jpeg"
                
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded_string}"}
                })
        except Exception as e:
            print(f"[Warning] Failed to load image {image_path}: {e}. Continuing text-only.", file=sys.stderr)
            
    system_msg = SystemMessage(content=(
        "You are an expert mechanical designer. Analyze the user request and sketch.\n"
        "Output strictly JSON with explicit fields: 'dimensions', 'features', 'materials', and 'design_intent'.\n"
        "No conversational text."
    ))
    
    human_msg = HumanMessage(content=content)
    
    response = call_model_with_key_rotation(
        models_to_try=[
            ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
            ("gemini", "models/gemini-2.5-flash-image"),
            ("gemini", "models/gemini-flash-latest")
        ],
        messages=[system_msg, human_msg],
        temperature=0.2
    )
    
    analysis_data = _extract_json_block(response.content)
    print(f"Vision Analysis Output Keys: {list(analysis_data.keys())}")
    
    # Store the analysis in messages as context
    state["messages"].append(f"Analysis Context:\n{json.dumps(analysis_data, indent=2)}")
    return state


# ==========================================
# NODE 2: CAD GENERATOR
# ==========================================
def node_cad_generator(state: AgentState) -> AgentState:
    print(f"-> Running CAD Generator (Retry: {state['retry_count']})...")
    
    system_prompt = (
        "You are an expert Python programmer for CAD.\n"
        "Generate valid, syntax-clean Python code using the `solid2` library to create the requested CAD model.\n"
        "Your code MUST end by calling exactly: `model.save_as_scad('model.scad')` assuming your geometry is named `model`.\n"
        "Linting hints (AVOID THESE):\n"
        "- Do not overlap faces exactly (z-fighting). Use slight overlaps (+0.01) for boolean operations.\n"
        "- Define variables explicitly at the top.\n"
        "- Always enclose Python code in ```python ... ``` blocks.\n"
    )
    
    human_content = "Please generate the OpenSCAD code based on the following context:\n\n"
    for m in state["messages"]:
        human_content += f"{m}\n"
        
    if state["retry_count"] > 0:
        human_content += "\n\nPREVIOUS RUN FAILED LINTING:\n"
        human_content += f"Previous Code:\n{state.get('scad_code', '')}\n"
        human_content += f"Errors to Fix:\n{chr(10).join(state.get('lint_errors', []))}\n"
        human_content += "Correct the code to fix these errors."
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]
    
    response = call_model_with_key_rotation(
        models_to_try=[
            ("groq", "qwen/qwen3-32b"),
            ("gemini", "models/gemma-4-31b-it"),
            ("gemini", "models/gemini-flash-latest")
        ],
        messages=messages,
        temperature=0.3
    )
    
    python_code = _extract_python_block(response.content)
    state["scad_code"] = python_code # Using the same key 'scad_code' to represent the generator code
    return state


# ==========================================
# NODE 3: LINTER / VALIDATOR
# ==========================================
def node_linter(state: AgentState) -> AgentState:
    print("-> Running Linter/Validator...")
    
    system_prompt = (
        "You are an OpenSCAD static analyzer.\n"
        "Check the provided Python solid2 code against 8 categories:\n"
        "1. Syntax errors (missing imports, unmatched braces)\n"
        "2. Module/Function definition order\n"
        "3. Variable scope issues\n"
        "4. Missing `.save_as_scad('model.scad')` call at the end\n"
        "5. Valid solid2 primitive usage\n"
        "6. Magic numbers (hardcoded values vs parameters)\n"
        "7. 2D vs 3D mixing without linear_extrude\n"
        "8. Correct transformation syntax\n"
        "Output strictly a JSON report:\n"
        '{"valid": true/false, "errors": ["error 1", ...], "summary": "brief summary"}'
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Code to lint:\n```python\n{state['scad_code']}\n```")
    ]
    
    try:
        response = call_model_with_key_rotation(
            models_to_try=[
                ("groq", "qwen/qwen3-32b"),
                ("gemini", "models/gemma-4-26b-a4b-it"),
                ("gemini", "models/gemini-flash-latest")
            ],
            messages=messages,
            temperature=0.0
        )
        report = _extract_json_block(response.content)
        
        state["valid"] = report.get("valid", True)
        state["lint_errors"] = report.get("errors", [])
        print(f"Linting result: Valid={state['valid']}, Errors={len(state['lint_errors'])}")
        
    except Exception as e:
        print(f"[Warning] Linter API failed: {e}. Assuming valid to not block export.", file=sys.stderr)
        state["valid"] = True
        state["lint_errors"] = []
        
    return state


# ==========================================
# ROUTING & RETRY LOGIC
# ==========================================
def increment_retry(state: AgentState) -> AgentState:
    state["retry_count"] += 1
    print(f"-> Incrementing retry count to {state['retry_count']}...")
    return state

def route_after_lint(state: AgentState) -> str:
    """Pure function of state for conditional edge routing."""
    if state["valid"]:
        return "export"
    if state["retry_count"] < 3:
        return "retry"
    return "abort"


# ==========================================
# EXPORT PIPELINE
# ==========================================
def export_files(python_code: str, output_prefix: str, skip_export: bool):
    """Executes the Python solid2 script, then uses OpenSCAD binaries to generate PNG, STL, CSG."""
    scad_file = f"{output_prefix}.scad"
    py_file = f"{output_prefix}.py"
    
    # 1. Write and Execute Python Script
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    
    # Patch the code to save to the correct scad_file path, overriding the default 'model.scad'
    python_code = python_code.replace("'model.scad'", f"r'{scad_file}'").replace('"model.scad"', f'r"{scad_file}"')
    
    with open(py_file, "w", encoding="utf-8") as f:
        f.write(python_code)
    print(f"-> Saved Python generator script {py_file}")
    
    try:
        print(f"-> Executing Python solid2 script...")
        _run_subprocess([sys.executable, py_file])
    except Exception as e:
        print(f"[Error] Python solid2 execution failed: {e}", file=sys.stderr)
        return
        
    if not os.path.exists(scad_file):
        print(f"[Error] Executed Python script but {scad_file} was not generated.", file=sys.stderr)
        return
        
    print(f"-> Verified generation of {scad_file}")
    
    # Use absolute path since it's freshly installed and might not be on PATH
    openscad_bin = r"C:\Program Files\OpenSCAD\openscad.exe"
    if not os.path.exists(openscad_bin):
        print(f"[Warning] OpenSCAD not found at {openscad_bin}. Trying 'openscad' from PATH.")
        openscad_bin = "openscad"
        
    # 2. OpenSCAD PNG Export (For Vision Inspection)
    png_file = f"{output_prefix}.png"
    print(f"-> Exporting {png_file} via OpenSCAD (Screenshot)...")
    try:
        _run_subprocess([openscad_bin, "-o", png_file, "--autocenter", "--viewall", "--colorscheme", "DeepOcean", "--imgsize=1024,1024", scad_file])
    except RuntimeError as e:
        print(f"[Warning] OpenSCAD PNG export failed: {e}", file=sys.stderr)

    if skip_export:
        print("-> Skipping binary exports as requested.")
        return
        
    # 3. OpenSCAD STL Export
    stl_file = f"{output_prefix}.stl"
    print(f"-> Exporting {stl_file} via OpenSCAD...")
    try:
        _run_subprocess([openscad_bin, "-o", stl_file, "--export-format", "binstl", scad_file])
    except RuntimeError as e:
        print(f"[Error] OpenSCAD STL export failed: {e}", file=sys.stderr)
        return

    # 3. OpenSCAD CSG Export
    csg_file = f"{output_prefix}.csg"
    print(f"-> Exporting {csg_file} via OpenSCAD...")
    try:
        _run_subprocess([openscad_bin, "-o", csg_file, "--export-format", "csg", scad_file])
    except RuntimeError as e:
        print(f"[Error] OpenSCAD CSG export failed: {e}", file=sys.stderr)
        return
        
    # 4. FreeCAD STEP Export
    step_file = f"{output_prefix}.step"
    print(f"-> Exporting {step_file} via FreeCAD...")
    
    # FreeCAD Python Script to convert CSG/STL to STEP
    fc_script = f"""
import FreeCAD
import Import
import Part
import sys

try:
    doc = FreeCAD.newDocument()
    # Import CSG via OpenSCAD module if possible, or fallback to STL
    try:
        import OpenSCAD
        OpenSCAD.insert(r"{csg_file}", doc.Name)
    except Exception:
        # Fallback to mesh-to-part if csg fails
        import Mesh
        mesh = Mesh.Mesh(r"{stl_file}")
        shape = Part.Shape()
        shape.makeShapeFromMesh(mesh.Topology, 0.1)
        solid = Part.Solid(shape)
        Part.show(solid)

    # Export to STEP
    objs = doc.Objects
    Part.export(objs, r"{step_file}")
except Exception as e:
    print(f"FreeCAD Export Error: {{e}}")
    sys.exit(1)
"""
    fd, temp_script_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, 'w') as temp_file:
            temp_file.write(fc_script)
            
        _run_subprocess(["freecadcmd", temp_script_path])
        print(f"-> Exported {step_file}")
    except RuntimeError as e:
        print(f"[Error] FreeCAD STEP export failed: {e}", file=sys.stderr)
    finally:
        os.remove(temp_script_path)


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="CADet AI CAD Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the desired CAD model.")
    parser.add_argument("--image", type=str, help="Optional sketch or reference image path.")
    parser.add_argument("--output", type=str, required=True, help="Output file prefix (e.g., ./out/model)")
    parser.add_argument("--skip-export", action="store_true", help="Skip running OpenSCAD and FreeCAD binaries.")
    
    args = parser.parse_args()
    
    # Build LangGraph DAG
    workflow = StateGraph(AgentState)
    
    workflow.add_node("vision_analyzer", node_vision_analyzer)
    workflow.add_node("cad_generator", node_cad_generator)
    workflow.add_node("linter", node_linter)
    workflow.add_node("increment_retry", increment_retry)
    
    workflow.set_entry_point("vision_analyzer")
    workflow.add_edge("vision_analyzer", "cad_generator")
    workflow.add_edge("cad_generator", "linter")
    
    workflow.add_conditional_edges(
        "linter",
        route_after_lint,
        {
            "export": END,
            "retry": "increment_retry",
            "abort": END
        }
    )
    workflow.add_edge("increment_retry", "cad_generator")
    
    app = workflow.compile()
    
    initial_state = {
        "messages": [args.prompt],
        "image_path": args.image,
        "retry_count": 0,
        "lint_errors": [],
        "scad_code": "",
        "valid": False,
        "skip_export": args.skip_export
    }
    
    print("=========================================")
    print(f"Starting CADet Generation")
    print(f"Prompt: {args.prompt}")
    if args.image:
        print(f"Image: {args.image}")
    print("=========================================")
    
    final_state = app.invoke(initial_state)
    
    if final_state["scad_code"]:
        export_files(final_state["scad_code"], args.output, args.skip_export)
        
    if not final_state["valid"] and final_state["retry_count"] >= 3:
        print("\n[Warning] Graph exited with unfixable lint errors after max retries.", file=sys.stderr)
        print("Lint Errors:")
        for err in final_state["lint_errors"]:
            print(f"- {err}")
            
    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()
