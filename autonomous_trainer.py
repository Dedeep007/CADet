import time
import os
import subprocess
import glob
import base64
import sys
from datetime import datetime
from typing import List, Any
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import random

load_dotenv()

def get_keys(prefix: str) -> List[str]:
    return [v for k, v in os.environ.items() if k.startswith(prefix) and v]

def vision_inspect_and_improve(png_path: str, original_prompt: str, scad_code: str) -> str:
    """Uses Gemini 2.5 Flash Image to visually inspect the rendered PNG and suggest improvements."""
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
                    f"Look at this rendered screenshot of the generated OpenSCAD model.\n"
                    f"1. Does it physically resemble the design goal?\n"
                    f"2. Are there obvious defects (missing holes, misaligned parts, intersecting solids that shouldn't intersect)?\n"
                    f"3. Are the proportions realistic for the physical world?\n\n"
                    f"Output a highly detailed, specific set of geometric improvements. Be critical! "
                    f"Your output will be used as the prompt for the next autonomous iteration."
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded_string}"}
        }
    ]
    
    system_msg = SystemMessage(content="You are an expert mechanical engineering AI vision inspector.")
    human_msg = HumanMessage(content=content)
    
    last_err = None
    
    # 1. Try Groq (Primary)
    from langchain_groq import ChatGroq
    g_keys = get_keys("GROQ_API_KEY")
    random.shuffle(g_keys)
    for key in g_keys:
        try:
            llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.3,
                max_tokens=1024,
                api_key=key
            )
            print(f"[Vision Inspector] Analyzing {png_path} with Groq...")
            response = llm.invoke([system_msg, human_msg])
            content_resp = response.content
            if isinstance(content_resp, list):
                content_resp = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content_resp)
            return f"Original Goal: {original_prompt}\n\nVision Inspector Feedback:\n{content_resp}\n\nIncorporate this feedback perfectly."
        except Exception as e:
            last_err = e
            
    # 2. Try Gemini (Fallback)
    keys = get_keys("GEMINI_API_KEY")
    if not keys and os.getenv("GOOGLE_API_KEY"):
        keys = [os.getenv("GOOGLE_API_KEY")]
    keys = [k for k in keys if k]
    random.shuffle(keys)
    for key in keys:
        try:
            llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-image",
                temperature=0.3,
                max_tokens=1024,
                google_api_key=key
            )
            print(f"[Vision Inspector] Analyzing {png_path} with Gemini...")
            response = llm.invoke([system_msg, human_msg])
            content_resp = response.content
            if isinstance(content_resp, list):
                content_resp = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content_resp)
            return f"Original Goal: {original_prompt}\n\nVision Inspector Feedback:\n{content_resp}\n\nIncorporate this feedback perfectly."
        except Exception as e:
            last_err = e
            
    print(f"[Vision Inspector Warning]: All keys failed. Last err: {last_err}")
    return original_prompt

BASE_PROMPTS = [
    "A print-in-place compliant mechanism gripper. It must have a central actuator that, when pushed, closes two flexible jaws. Include mounting holes.",
    "A complex planetary gear system. 1 sun gear, 3 planet gears, and 1 outer ring gear. They must mesh perfectly with realistic tolerances.",
    "A drone frame optimized for 5-inch props with exact mounting holes for a 30.5x30.5mm flight controller stack. Needs structural cross-bracing."
]

def main():
    os.makedirs("experiments", exist_ok=True)
    with open(".gitignore", "a") as f:
        f.write("\nexperiments/\n*.png\n*.stl\n*.csg\n*.step\n")
        
    print("Starting Autonomous CAD Trainer Loop (10-Hour Goal)")
    
    iteration = 0
    while True:
        base_prompt = BASE_PROMPTS[iteration % len(BASE_PROMPTS)]
        current_prompt = base_prompt
        
        # 3 refinement loops per base prompt
        for refinement_step in range(3):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = f"experiments/{timestamp}_iter{iteration}_step{refinement_step}"
            os.makedirs(exp_dir, exist_ok=True)
            output_prefix = f"{exp_dir}/model"
            
            print(f"\n==========================================")
            print(f"Iteration {iteration}, Refinement Step {refinement_step}")
            print(f"Outputting to: {output_prefix}")
            print(f"==========================================")
            
            cmd = [
                ".\\venv\\Scripts\\python", "cad_agent.py",
                "--prompt", current_prompt,
                "--output", output_prefix
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print("[Trainer] cad_agent.py exited with an error. Continuing to next step...")
                time.sleep(10)
                continue
                
            png_path = f"{output_prefix}.png"
            scad_path = f"{output_prefix}.scad"
            
            scad_code = ""
            if os.path.exists(scad_path):
                with open(scad_path, "r") as f:
                    scad_code = f.read()
                    
            if refinement_step < 2:
                # Inspect and generate new prompt
                current_prompt = vision_inspect_and_improve(png_path, base_prompt, scad_code)
                print(f"\n[Trainer] New improved prompt generated for next step.")
            
            time.sleep(20) # Avoid extreme rate limiting
            
        iteration += 1

if __name__ == "__main__":
    main()
