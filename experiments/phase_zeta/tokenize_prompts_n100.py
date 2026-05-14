"""Tokenize 50 NEW natural-language prompts for the N=100 settling battery.

These extend the existing N=50 set (tokenize_prompts.py) with another 50
diverse prompts. The new set is structurally similar — varied lengths,
domains, and registers — but uses fresh content so the two halves can
be analyzed independently OR pooled as N=100.

Distribution targets (matching the existing 50):
  - 4 Q&A
  - 4 definitions
  - 4 continuations
  - 2 math
  - 2 reasoning/factual
  - 2 instructions
  - 2 dialogue
  - 5 long-form descriptive
  - 6 code (Python, JS, SQL, Bash, ...)
  - 2 poetry
  - 5 technical (ML, physics, chemistry, biology, CS)
  - 2 errors
  - 2 idioms
  - 6 logical (hypothesis/negation/quantifier/temporal/conditional/causal)
  - 2 history/geography/factual

Run: python experiments/phase_zeta/tokenize_prompts_n100.py
"""
from transformers import AutoTokenizer

REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"

NEW_PROMPTS = [
    # Q&A (4)
    ("q_capital_egypt",     "What is the capital of Egypt?"),
    ("q_speed_of_light",    "What is the approximate speed of light in a vacuum?"),
    ("q_wrote_iliad",       "Who is the traditional author credited with writing the Iliad?"),
    ("q_dna_full",          "What does DNA stand for?"),

    # Definitions (4)
    ("def_osmosis",         "Osmosis is the process in which water molecules"),
    ("def_inflation",       "Inflation in economics refers to the general"),
    ("def_algorithm",       "An algorithm is a finite sequence of"),
    ("def_metaphor",        "A metaphor is a figure of speech that"),

    # Continuations (4)
    ("cont_old_man",        "The old man stared out at the sea, remembering the day when"),
    ("cont_letter",         "My dearest Eleanor, by the time you read these words I will"),
    ("cont_lab",            "The reaction proceeded slowly at first, but then"),
    ("cont_garden",         "She walked through the garden gate and saw, to her surprise,"),

    # Math (2)
    ("math_div",            "Twenty divided by four equals"),
    ("math_word_problem",   "If a train leaves at noon traveling 60 miles per hour, how far does it travel by"),

    # Reasoning / factual (2)
    ("reasoning_seasons",   "The seasons on Earth are caused primarily by"),
    ("reasoning_eclipse",   "A solar eclipse occurs when the moon passes"),

    # Instructions (2)
    ("instr_explain",       "Explain in simple terms how a refrigerator keeps food cold:"),
    ("instr_list",          "List three benefits of regular exercise:"),

    # Dialogue (2)
    ("dialog_negotiate",    "Buyer: I'm not sure the price is right.\nSeller: I can offer you a"),
    ("dialog_interview",    "Interviewer: Tell me about a time you faced a difficult challenge.\nCandidate:"),

    # Long-form descriptive (5)
    ("long_market",         "The bustling marketplace was alive with the sounds of merchants haggling, children laughing, and the rhythmic clang of"),
    ("long_storm",          "Heavy storm clouds gathered on the horizon, and the wind, which had been gentle moments before, now"),
    ("long_research",       "After three years of research and countless hours in the laboratory, the team finally arrived at a result that"),
    ("long_journey",        "The journey across the mountains took longer than anyone had anticipated, mainly because the unexpected snowfall in"),
    ("long_meeting",        "At the company-wide quarterly meeting, the chief executive announced that, beginning next fiscal year, all departments would"),

    # Code (6)
    ("code_js_arrow",       "const sum = (a, b) =>"),
    ("code_python_dict",    "user_info = {\n    'name': 'Alice',\n    'age': 30,\n    'roles':"),
    ("code_bash_loop",      "for file in *.txt; do\n    echo \"Processing $file\"\n    cat \"$file\" |"),
    ("code_html_tag",       "<div class=\"container\">\n    <h1>Welcome</h1>\n    <p>This is"),
    ("code_sql_join",       "SELECT u.name, o.total FROM users u JOIN orders o ON u.id ="),
    ("code_rust_fn",        "fn factorial(n: u64) -> u64 {\n    if n <= 1 { 1 } else {"),

    # Poetry (2)
    ("poetry_blake",        "Tyger Tyger, burning bright,\nIn the forests of the night;\nWhat immortal hand or eye\nCould frame thy"),
    ("poetry_dickinson",    "Hope is the thing with feathers\nThat perches in the soul,\nAnd sings the tune without"),

    # Technical (5)
    ("tech_quantum",        "Quantum entanglement is a phenomenon in which the quantum states of two particles become"),
    ("tech_compiler",       "A compiler translates source code from a high-level programming language to a"),
    ("tech_protein",        "Proteins are large biomolecules consisting of one or more long chains of"),
    ("tech_database_index", "A database index is a data structure that improves the speed of data retrieval operations on a database table at the cost of"),
    ("tech_neural",         "A convolutional neural network is particularly effective at image recognition because"),

    # Errors (2)
    ("error_segfault",      "Segmentation fault (core dumped). The program attempted to access memory at"),
    ("error_typeerror",     "TypeError: unsupported operand type(s) for +: 'int' and"),

    # Idioms (2)
    ("idiom_horse",         "Don't put the cart before the horse means"),
    ("idiom_apple",         "An apple a day keeps the doctor away suggests that"),

    # Logical structures (6)
    ("logic_hypothesis2",   "If global temperatures continue to rise at the current rate, scientists predict that"),
    ("logic_negation2",     "Contrary to what some critics have claimed, recent studies show that"),
    ("logic_quantifier2",   "Most of the participants in the survey reported that they would prefer to"),
    ("logic_temporal2",     "Before the invention of the printing press, books were"),
    ("logic_conditional2",  "Whenever the temperature drops below freezing, the pipes in the basement"),
    ("logic_causal2",       "Due to a sudden shift in consumer preferences, the company was forced to"),

    # History / geography / factual (2)
    ("history_moon",        "The first human to walk on the moon was"),
    ("geography_desert",    "The largest hot desert on Earth, by area, is the"),
]

assert len(NEW_PROMPTS) == 50, f"expected 50 new prompts, got {len(NEW_PROMPTS)}"


def main():
    tok = AutoTokenizer.from_pretrained(REPO)
    # Verify BOS token id
    bos_id = tok.bos_token_id
    print(f"# tokenizer: {REPO}")
    print(f"# bos_token_id = {bos_id}")
    print()
    print("NEW_PROMPTS = {")
    for label, text in NEW_PROMPTS:
        ids = tok.encode(text, add_special_tokens=True)
        # Sanity: every prompt should start with BOS=128000 (Llama-style)
        assert ids[0] == 128000, f"{label} missing BOS=128000: starts {ids[:3]}"
        print(f'    "{label}":'.ljust(28) + f'"{",".join(str(t) for t in ids)}",'
              + f"  # n={len(ids)}")
    print("}")


if __name__ == "__main__":
    main()
