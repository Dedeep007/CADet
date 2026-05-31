import os
import time
import subprocess
import base64
import sys
from datetime import datetime
from dotenv import load_dotenv

from architect_agent import run_architect
from component_agent import run_component_engineer
from assembler_agent import run_assembler

# For Vision Inspector
from langchain_core.messages import HumanMessage, SystemMessage
from llm_utils import call_model_with_key_rotation

load_dotenv()

def vision_inspect_and_improve(png_path: str, original_prompt: str, scad_code: str) -> str:
    """Uses LLM to visually inspect the rendered PNG and suggest improvements."""
    if not os.path.exists(png_path):
        return original_prompt + "\n\n(Previous run failed to generate a 3D render. Please fix syntax and geometry.)"
        
    try:
        with open(png_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return original_prompt
        
    content = [
        {
            "type": "text", 
            "text": f"Original Design Goal: {original_prompt}\n\n"
                    f"Look at this rendered screenshot of the generated CadQuery assembly.\n"
                    f"1. Does it physically resemble the design goal?\n"
                    f"2. Are there obvious defects (missing holes, misaligned parts, intersecting solids that shouldn't intersect)?\n"
                    f"3. Are the proportions realistic for the physical world?\n\n"
                    f"Output a highly detailed, specific set of geometric improvements. Be critical! "
                    f"Your output will be used by the Architect for the next iteration."
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded_string}"}
        }
    ]
    
    system_msg = SystemMessage(content="You are an expert mechanical engineering AI vision inspector.")
    human_msg = HumanMessage(content=content)
    
    models_to_try = [
        ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
        ("gemini", "models/gemini-2.5-flash-image")
    ]
    
    try:
        response = call_model_with_key_rotation(models_to_try, [system_msg, human_msg], temperature=0.3)
        content_resp = response.content if hasattr(response, 'content') else str(response)
        if isinstance(content_resp, list):
            content_resp = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content_resp)
        return f"Original Goal: {original_prompt}\n\nVision Inspector Feedback:\n{content_resp}\n\nIncorporate this feedback perfectly."
    except Exception as e:
        print(f"[Vision Inspector Warning]: All keys failed. Last err: {e}")
        return original_prompt

BASE_PROMPTS = [
    "A complex planetary gear system. 1 sun gear, 3 planet gears, and 1 outer ring gear. They must mesh perfectly with realistic tolerances.",
    "A print-in-place compliant mechanism gripper. It must have a central actuator that, when pushed, closes two flexible jaws. Include mounting holes.",
    "A drone frame optimized for 5-inch props with exact mounting holes for a 30.5x30.5mm flight controller stack. Needs structural cross-bracing."
]

def export_files(stl_file: str, base_name: str):
    """Exports PNG from STL using OpenSCAD as a rendering engine."""
    if not os.path.exists(stl_file):
        print(f"[Error] {stl_file} does not exist.")
        return
        
    png_file = f"{base_name}.png"
    
    # Create a temporary wrapper scad file that imports the STL
    wrapper_scad = f"{base_name}_render.scad"
    with open(wrapper_scad, "w", encoding="utf-8") as f:
        f.write(f'import("{os.path.basename(stl_file)}");\n')
    
    # Export PNG
    print(f"-> Exporting {png_file} via OpenSCAD (Screenshot)...")
    cmd_png = ["openscad", "-o", png_file, "--autocenter", "--viewall", "--colorscheme", "Tomorrow Night", wrapper_scad]
    try:
        subprocess.run(cmd_png, capture_output=True, text=True, check=True, shell=(sys.platform == "win32"))
    except subprocess.CalledProcessError as e:
        print(f"[Warning] OpenSCAD PNG export failed: {e.stderr}", file=sys.stderr)
        
    if os.path.exists(wrapper_scad):
        os.remove(wrapper_scad)

def main():
    os.makedirs("experiments", exist_ok=True)
    print("Starting Multi-Agent Distributed CAD Trainer Loop")
    
    iteration = 0
    while True:
        base_prompt = BASE_PROMPTS[iteration % len(BASE_PROMPTS)]
        current_prompt = base_prompt
        
        for refinement_step in range(3):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = f"experiments/{timestamp}_iter{iteration}_step{refinement_step}"
            os.makedirs(exp_dir, exist_ok=True)
            output_prefix = f"{exp_dir}/model"
            
            print(f"\n==========================================")
            print(f"Iteration {iteration}, Refinement Step {refinement_step}")
            print(f"Outputting to: {output_prefix}")
            print(f"==========================================")
            
            # 1. The Architect
            vision_feedback = current_prompt if refinement_step > 0 else ""
            architecture = run_architect(base_prompt, vision_feedback)
            
            # 2. The Component Engineers
            component_names = []
            for comp in architecture.get("components", []):
                name = comp.get("name", "part").replace(" ", "_").lower()
                constraints = comp.get("constraints", [])
                
                # Write component to file
                comp_code = run_component_engineer(name, constraints)
                comp_file = os.path.join(exp_dir, f"{name}.py")
                with open(comp_file, "w", encoding="utf-8") as f:
                    f.write(comp_code)
                    
                component_names.append(name)
                
            # 3. The Assembler
            instructions = architecture.get("assembly_instructions", [])
            assembly_code = run_assembler(architecture.get("assembly_name", "assembly"), component_names, instructions)
            assembly_file = os.path.join(exp_dir, "assembly.py")
            with open(assembly_file, "w", encoding="utf-8") as f:
                f.write(assembly_code)
                
            # 4. Generate the models (CadQuery script saves .stl)
            stl_path = f"{output_prefix}.stl"
            print(f"-> Executing CadQuery Assembly Script...")
            
            try:
                subprocess.run([sys.executable, "assembly.py"], cwd=exp_dir, capture_output=True, text=True, check=True)
                print("-> Verified generation of CAD models.")
                export_files(stl_path, output_prefix)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Assembly Execution failed: {e.stderr}", file=sys.stderr)
            
            # 5. Vision Inspection
            png_path = f"{output_prefix}.png"
            if refinement_step < 2:
                current_prompt = vision_inspect_and_improve(png_path, base_prompt, assembly_code)
                print(f"\n[Trainer] New improved feedback generated for Architect.")
            
            time.sleep(10) # API pacing
            
        iteration += 1

if __name__ == "__main__":
    main()
