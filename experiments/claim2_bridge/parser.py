"""Tiny recursive-descent parser for arithmetic expressions.

Grammar:
  expr   := term (('+' | '-') term)*
  term   := factor ('*' factor)*
  factor := identifier | integer | '(' expr ')' | '-' factor   (unary minus)

Returns an AST as nested tuples. Tuple shapes:
  ('var', name)
  ('const', n)
  ('add', e1, e2, ..., eN)    # n-ary after later flattening
  ('sub', e1, e2)             # binary; non-commutative
  ('mul', e1, e2, ..., eN)
  ('neg', e)                  # unary minus
"""
from __future__ import annotations

import re
from typing import Tuple

Tok = Tuple[str, str]


def tokenize(s: str) -> list[Tok]:
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(("INT", s[i:j]))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(("ID", s[i:j]))
            i = j
            continue
        if c in "+-*()":
            tokens.append(("OP", c))
            i += 1
            continue
        raise ValueError(f"unexpected char {c!r} at pos {i} in {s!r}")
    return tokens


class Parser:
    def __init__(self, tokens: list[Tok]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Tok | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, kind: str, value: str | None = None) -> Tok:
        t = self.peek()
        if t is None or t[0] != kind or (value is not None and t[1] != value):
            raise ValueError(f"expected {kind}{'='+value if value else ''}, got {t!r}")
        self.pos += 1
        return t

    def parse_expr(self):
        left = self.parse_term()
        while True:
            t = self.peek()
            if t is None or t[0] != "OP" or t[1] not in ("+", "-"):
                break
            op = t[1]; self.pos += 1
            right = self.parse_term()
            left = ("add" if op == "+" else "sub", left, right)
        return left

    def parse_term(self):
        left = self.parse_factor()
        while True:
            t = self.peek()
            if t is None or t[0] != "OP" or t[1] != "*":
                break
            self.pos += 1
            right = self.parse_factor()
            left = ("mul", left, right)
        return left

    def parse_factor(self):
        t = self.peek()
        if t is None:
            raise ValueError("unexpected end")
        if t[0] == "OP" and t[1] == "-":
            self.pos += 1
            return ("neg", self.parse_factor())
        if t[0] == "OP" and t[1] == "(":
            self.pos += 1
            e = self.parse_expr()
            self.eat("OP", ")")
            return e
        if t[0] == "INT":
            self.pos += 1
            return ("const", int(t[1]))
        if t[0] == "ID":
            self.pos += 1
            return ("var", t[1])
        raise ValueError(f"unexpected token {t!r}")


def parse(s: str):
    p = Parser(tokenize(s))
    ast = p.parse_expr()
    if p.peek() is not None:
        raise ValueError(f"trailing tokens after parse: {p.tokens[p.pos:]!r}")
    return ast


if __name__ == "__main__":
    tests = [
        "x + y",
        "x*y + z",
        "(x + y) * (x - y)",
        "-(x + 1)",
        "2 + 3 * x",
    ]
    for s in tests:
        print(f"{s!r:30s} -> {parse(s)}")
