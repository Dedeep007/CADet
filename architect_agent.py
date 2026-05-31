from langchain_core.messages import SystemMessage, HumanMessage
from llm_utils import call_model_with_key_rotation, _extract_json_block
import json
import sys

def run_architect(design_prompt: str, vision_feedback: str = "") -> dict:
    """
    Acts as the Master Planner. 
    Outputs a strict JSON specification breaking down the assembly into sub-components.
    """
    print("-> Running Architect Agent...")
    system_msg = SystemMessage(content=(
        "You are an expert mechanical engineering Architect Agent.\n"
        "Your job is to break down a user's complex CAD design prompt into a modular assembly of sub-components.\n"
        "You do NOT write code. You output a strict JSON schema that other AI agents will use to write code.\n"
        "For each component, provide explicit, mathematical constraints (dimensions, tolerances, hole diameters, module, etc.).\n"
        "If there are mating parts, ensure the dimensions allow them to fit perfectly (e.g. peg diameter = 9.8mm, hole diameter = 10.0mm).\n"
        "\n"
        "Output ONLY valid JSON with this exact schema:\n"
        "{\n"
        "  \"assembly_name\": \"string\",\n"
        "  \"components\": [\n"
        "    {\n"
        "      \"name\": \"string (e.g. sun_gear)\",\n"
        "      \"description\": \"string\",\n"
        "      \"constraints\": [\"string constraint 1\", \"string constraint 2\"]\n"
        "    }\n"
        "  ],\n"
        "  \"assembly_instructions\": [\"string instruction 1\"]\n"
        "}\n"
    ))
    
    prompt = f"User Request: {design_prompt}"
    if vision_feedback:
        prompt += f"\n\nPrevious QA Feedback to fix in the architecture:\n{vision_feedback}"
        
    human_msg = HumanMessage(content=prompt)
    
    # We use qwen3-32b or gemma for reasoning
    models_to_try = [
        ("groq", "qwen/qwen3-32b"),
        ("gemini", "models/gemma-4-31b-it"),
        ("gemini", "models/gemini-flash-latest")
    ]
    
    try:
        response = call_model_with_key_rotation(models_to_try, [system_msg, human_msg], temperature=0.3, max_tokens=2048)
        spec = _extract_json_block(response)
        if not spec or "components" not in spec:
            print("[Warning] Architect failed to return valid schema. Returning fallback.", file=sys.stderr)
            return _get_fallback_schema(design_prompt)
        print(f"-> Architect defined {len(spec['components'])} components.")
        return spec
    except Exception as e:
        print(f"[Error] Architect Agent failed: {e}", file=sys.stderr)
        return _get_fallback_schema(design_prompt)

def _get_fallback_schema(prompt: str) -> dict:
    return {
        "assembly_name": "fallback_assembly",
        "components": [
            {
                "name": "main_body",
                "description": f"The entire design based on: {prompt}",
                "constraints": ["Model the entire requested object."]
            }
        ],
        "assembly_instructions": ["Simply combine the parts if any."]
    }

if __name__ == "__main__":
    # Test the architect
    print(json.dumps(run_architect("A simple box with a lid."), indent=2))
