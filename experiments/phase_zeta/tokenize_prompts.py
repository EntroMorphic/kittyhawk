"""Tokenize a set of natural-language prompts using the BitNet tokenizer.

Outputs token ID sequences in the format eviction_battery.py expects:
  PROMPTS = { "label": "1,1841,...", ... }

Run with: python experiments/phase_zeta/tokenize_prompts.py
"""
from transformers import AutoTokenizer

REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"

# 20-prompt diverse battery for plan B red-team. Mix:
# - factual questions (capital_france-style)
# - definitions and explanations
# - continuations
# - math
# - common phrases / idioms
# - reasoning
# - dialog / instruction
# - long descriptive
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
