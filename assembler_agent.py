from langchain_core.messages import SystemMessage, HumanMessage
from llm_utils import call_model_with_key_rotation, _extract_python_block
import sys

def run_assembler(assembly_name: str, component_names: list, assembly_instructions: list, previous_errors: str = "") -> str:
    """
    Acts as the System Integrator. 
    Writes a master assembly.py script that imports and positions the components.
    """
    print(f"-> Running Assembler Agent for {assembly_name}...")
    system_msg = SystemMessage(content=(
        "You are an expert Python programmer for CAD assemblies.\n"
        "Your job is to write a master `assembly.py` script that combines several solid2 components.\n"
        "You will be given a list of component filenames and assembly instructions.\n"
        "Each component file has a `get_component()` function.\n"
        "\n"
        "Example output:\n"
        "```python\n"
        "from solid2 import *\n"
        "import sun_gear\n"
        "import planet_gear\n"
        "\n"
        "def main():\n"
        "    sun = sun_gear.get_component()\n"
        "    planet1 = planet_gear.get_component().translate([10, 0, 0])\n"
        "    assembly = sun + planet1\n"
        "    assembly.save_as_scad('model.scad')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
        "```\n"
        "Output ONLY the python code."
    ))
    
    prompt = f"Components to import: {', '.join(component_names)}\nInstructions:\n"
    for ins in assembly_instructions:
        prompt += f"- {ins}\n"
        
    if previous_errors:
        prompt += f"\nPrevious syntax error to fix:\n{previous_errors}"
        
    human_msg = HumanMessage(content=prompt)
    
    models_to_try = [
        ("groq", "qwen/qwen3-32b"),
        ("gemini", "models/gemma-4-31b-it"),
        ("gemini", "models/gemini-flash-latest")
    ]
    
    try:
        response = call_model_with_key_rotation(models_to_try, [system_msg, human_msg], temperature=0.2, max_tokens=2048)
        code = _extract_python_block(response)
        return code
    except Exception as e:
        print(f"[Error] Assembler Agent failed: {e}", file=sys.stderr)
        return ""

if __name__ == "__main__":
    code = run_assembler("gearbox", ["sun_gear", "planet_gear"], ["Translate planet gear by 20mm in X."])
    print(code)
