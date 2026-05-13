"""Tokenize a set of natural-language prompts using the BitNet tokenizer.

Outputs token ID sequences in the format eviction_battery.py expects:
  PROMPTS = { "label": "1,1841,...", ... }

Run with: python experiments/phase_zeta/tokenize_prompts.py
"""
from transformers import AutoTokenizer

REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"

# 50-prompt diverse battery for plan B settlement (Track B of
# glyph_gaps_2026-05-13_synthesize.md). Original 20 + 30 new across
# code, poetry, dialog, technical jargon, error messages, multi-turn.
PROMPTS = [
    ("q_capital_france",   "What is the capital of France?"),
    ("q_capital_japan",    "What is the capital of Japan?"),
    ("q_largest_planet",   "What is the largest planet in our solar system?"),
    ("q_who_hamlet",       "Who wrote the play Hamlet?"),
    ("def_photosynth",     "Photosynthesis is the process by which"),
    ("def_gravity",        "Gravity is a fundamental force that"),
    ("cont_once",          "Once upon a time, in a kingdom far away,"),
    ("cont_dark_stormy",   "It was a dark and stormy night when the"),
    ("math_add",           "12 plus 7 equals"),
    ("math_mul",           "Five times eight equals"),
    ("color_sky",          "The color of the sky on a clear day is"),
    ("reasoning_water",    "Water boils at 100 degrees Celsius at"),
    ("instr_translate",    "Translate the following to French: Hello, how are you?"),
    ("instr_summary",      "In one sentence, summarize what photosynthesis means:"),
    ("dialog_greet",       "A: Hello, how are you today? B:"),
    ("idiom_break_ice",    "The phrase 'break the ice' means"),
    ("long_desc_forest",   "Deep within the ancient forest, where sunlight barely reached the moss-covered ground,"),
    ("long_lab",           "The scientist carefully adjusted the microscope, examining the slide for any sign of"),
    ("long_recipe",        "To make a perfect omelet, first crack two large eggs into a bowl and whisk them together with"),
    ("long_argument",      "Although critics have argued that the policy is too costly, supporters maintain that"),
    # +30 to reach N=50
    ("code_python_fn",     "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"),
    ("code_loop",          "for i in range(10):\n    print("),
    ("code_class",         "class Animal:\n    def __init__(self, name):\n        self.name ="),
    ("code_import",        "import numpy as np\n\ndef mean(arr):\n    return"),
    ("code_sql",           "SELECT name, age FROM users WHERE age > 18 ORDER BY"),
    ("poetry_haiku",       "Cherry blossoms fall\nSoftly on the quiet pond"),
    ("poetry_iambic",      "Shall I compare thee to a summer's day?\nThou art more lovely and more"),
    ("dialog_qa",          "Q: How do plants get energy?\nA: Plants get energy through"),
    ("dialog_multi",       "Alice: I've been thinking about that book.\nBob: Which one?\nAlice:"),
    ("technical_ml",       "A transformer is a neural network architecture that uses self-attention to"),
    ("technical_physics",  "The second law of thermodynamics states that"),
    ("technical_chem",     "Sodium chloride dissolves in water because"),
    ("error_traceback",    "Traceback (most recent call last):\n  File \"app.py\", line 12, in <module>\n    result = compute("),
    ("error_message",      "Error: Cannot read property 'name' of undefined"),
    ("instruct_step",      "Step 1: Open the package.\nStep 2: Remove the protective film.\nStep 3:"),
    ("instruct_recipe",    "First, preheat the oven to 350 degrees. Then,"),
    ("hypothesis",         "If the cost of solar panels continues to fall, then"),
    ("comparison",         "Unlike traditional methods, which rely on manual review, this approach"),
    ("negation",           "It is not the case that all birds can fly; for example,"),
    ("quantifier",         "Every student in the class submitted their assignment except for"),
    ("temporal",           "Yesterday I went to the store, today I am working from home, and tomorrow I will"),
    ("conditional",        "If it rains tomorrow, we will need to"),
    ("causal",             "Because the engine overheated, the car"),
    ("definition_term",    "Machine learning is a subset of artificial intelligence that involves"),
    ("history_fact",       "World War II ended in"),
    ("geography_river",    "The longest river in the world is the"),
    ("biology_cell",       "The mitochondria of a cell are responsible for"),
    ("idiom_spill",        "When she said 'spill the beans,' she meant"),
    ("idiom_back",         "He said it was a piece of cake, meaning that the task was"),
    ("longform_essay",     "The Industrial Revolution fundamentally transformed human society through three primary mechanisms: first, the mechanization of production processes; second,"),
]


def main():
    tok = AutoTokenizer.from_pretrained(REPO)
    print("PROMPTS = {")
    for label, text in PROMPTS:
        ids = tok.encode(text, add_special_tokens=True)
        print(f'    "{label}":'.ljust(28) + f'"{",".join(str(t) for t in ids)}",'
              + f"  # n={len(ids)}: {text!r}")
    print("}")


if __name__ == "__main__":
    main()
