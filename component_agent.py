import os
import subprocess
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from llm_utils import call_model_with_key_rotation, _extract_python_block
import sys

class ComponentState(TypedDict):
    component_name: str
    constraints: List[str]
    messages: List[str]
    scad_code: str
    retry_count: int
    lint_errors: List[str]

def node_generator(state: ComponentState) -> ComponentState:
    print(f"  -> Generating code for {state['component_name']} (Retry {state['retry_count']})")
    system_prompt = (
        "You are an expert Python programmer for CAD.\n"
        "Generate valid, syntax-clean Python code using the `solid2` library to create the requested component.\n"
        "Your code MUST end by defining a function `get_component()` that returns the solid2 geometry object.\n"
        "DO NOT call `save_as_scad`. The assembler agent will handle saving.\n"
        "Example:\n"
        "```python\n"
        "from solid2 import *\n\n"
        "def get_component():\n"
        "    return cube(10)\n"
        "```\n"
        "CRITICAL: Keep your internal <think> reasoning EXTREMELY brief (under 100 words).\n"
        "Linting hints: Define variables explicitly. Avoid z-fighting."
    )
    
    human_content = f"Design Component: {state['component_name']}\nConstraints:\n"
    for c in state['constraints']:
        human_content += f"- {c}\n"
        
    for m in state["messages"]:
        human_content += f"\n{m}"
        
    if state["retry_count"] > 0:
        human_content += f"\n\nPREVIOUS CODE FAILED LINTING:\nPrevious Code:\n{state.get('scad_code', '')}\nErrors:\n{chr(10).join(state.get('lint_errors', []))}\nFix these errors."
        
    models_to_try = [
        ("groq", "qwen/qwen3-32b"),
        ("gemini", "models/gemma-4-31b-it"),
        ("gemini", "models/gemini-flash-latest")
    ]
    
    try:
        response = call_model_with_key_rotation(models_to_try, [SystemMessage(content=system_prompt), HumanMessage(content=human_content)], temperature=0.3)
        python_code = _extract_python_block(response)
        state["scad_code"] = python_code
    except Exception as e:
        print(f"  [Error] Generator failed: {e}", file=sys.stderr)
        state["scad_code"] = ""
        
    return state

def node_linter(state: ComponentState) -> ComponentState:
    code = state.get("scad_code", "")
    errors = []
    
    if not code:
        errors.append("Empty code provided.")
    elif "def get_component():" not in code:
        errors.append("Missing required function definition: def get_component():")
    elif "save_as_scad" in code:
        errors.append("Do not call save_as_scad(). Just return the geometry.")
        
    if not errors:
        # Syntax check
        test_file = f".temp_{state['component_name']}.py"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(code + "\n\n# Syntax Check\ntry:\n    geom = get_component()\n    assert geom is not None\nexcept Exception as e:\n    print(e)\n    exit(1)")
            
        try:
            res = subprocess.run(["python", test_file], capture_output=True, text=True, timeout=10)
            if res.returncode != 0:
                errors.append(f"Execution Error: {res.stderr.strip() or res.stdout.strip()}")
        except Exception as e:
            errors.append(f"Subprocess Error: {e}")
            
        if os.path.exists(test_file):
            os.remove(test_file)
            
    state["lint_errors"] = errors
    if errors:
        state["retry_count"] += 1
        
    return state

def route_lint(state: ComponentState):
    if not state["lint_errors"] or state["retry_count"] >= 3:
        return "end"
    return "generate"

def run_component_engineer(name: str, constraints: List[str], previous_errors: str = "") -> str:
    """Generates and lints python solid2 code for a single component."""
    workflow = StateGraph(ComponentState)
    workflow.add_node("generate", node_generator)
    workflow.add_node("lint", node_linter)
    workflow.set_entry_point("generate")
    workflow.add_conditional_edges("lint", route_lint, {"generate": "generate", "end": END})
    app = workflow.compile()
    
    messages = []
    if previous_errors:
        messages.append(f"Feedback from previous assembly failure: {previous_errors}")
        
    initial_state = ComponentState(
        component_name=name,
        constraints=constraints,
        messages=messages,
        scad_code="",
        retry_count=0,
        lint_errors=[]
    )
    
    print(f"-> Starting Component Engineer: {name}")
    final_state = app.invoke(initial_state)
    
    if final_state["lint_errors"]:
        print(f"  [Warning] Component {name} finished with unfixable lint errors.")
    else:
        print(f"  -> Component {name} successfully generated.")
        
    return final_state["scad_code"]

if __name__ == "__main__":
    code = run_component_engineer("test_cube", ["A 10x10x10 cube", "Centered"])
    print(code)
