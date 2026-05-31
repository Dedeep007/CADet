from autonomous_trainer import BASE_PROMPTS
from architect_agent import run_architect
from component_agent import run_component_engineer
from assembler_agent import run_assembler
import os

os.makedirs('experiments/test_multiagent', exist_ok=True)
arch = run_architect(BASE_PROMPTS[0])
print(arch)

names = []
for c in arch.get('components', []):
    name = c.get("name").replace(" ", "_").lower()
    constraints = c.get("constraints", [])
    code = run_component_engineer(name, constraints)
    with open(f'experiments/test_multiagent/{name}.py', 'w', encoding="utf-8") as f:
        f.write(code)
    names.append(name)

assembly_code = run_assembler(arch.get('assembly_name', 'test_assembly'), names, arch.get('assembly_instructions', []))
with open(f'experiments/test_multiagent/assembly.py', 'w', encoding="utf-8") as f:
    f.write(assembly_code)

print("Test complete.")
