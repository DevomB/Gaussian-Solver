"""
Gaussian elimination to RREF, with two different strategies for choosing
row operations.

    machine  Textbook partial pivoting. Pick the largest available pivot,
             scale it to 1 immediately, then eliminate. Fewest decisions,
             fewest steps, but fractions appear early and stay.

    human    How a person actually does it on paper. Hunt for a pivot of
             1 or -1, refuse to divide until the very end, keep every
             intermediate entry a whole number, and cancel common factors
             out of a row whenever they show up.

Both produce identical, exact answers. They differ in how ugly the
scratch work gets along the way.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd

Matrix = list


# ---------------------------------------------------------------- display


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    DIM = "\033[2m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def disable_colors():
    for attr in dir(Colors):
        if not attr.startswith("__"):
            setattr(Colors, attr, "")


if not sys.stdout.isatty():
    disable_colors()


def fmt(val):
    """Render a Fraction the way you would write it by hand."""
    if val.denominator == 1:
        return str(val.numerator)
    return "%d/%d" % (val.numerator, val.denominator)


def fmt_coeff(val):
    """Render a multiplier, dropping a redundant 1 so it reads as 3R1."""
    if val == 1:
        return ""
    if val == -1:
        return "-"
    return fmt(val)


def format_matrix(matrix, is_augmented=True):
    if not matrix:
        return "(empty)"
    num_cols = len(matrix[0])
    widths = []
    for c in range(num_cols):
        widths.append(max(len(fmt(matrix[r][c])) for r in range(len(matrix))))
    lines = []
    for row in matrix:
        cells = []
        for c, v in enumerate(row):
            cells.append(fmt(v).rjust(widths[c]))
        if is_augmented and num_cols > 1:
            lines.append("[  " + "  ".join(cells[:-1]) + "  |  " + cells[-1] + "  ]")
        else:
            lines.append("[  " + "  ".join(cells) + "  ]")
    return "\n".join(lines)


# ------------------------------------------------------------------ trace


@dataclass
class Step:
    """One row operation, plus the matrix as it looked afterwards."""

    description: str
    matrix: list
    note: str = ""
    has_fractions: bool = False


@dataclass
class Trace:
    steps: list = field(default_factory=list)
    swaps: int = 0
    scalings: int = 0
    eliminations: int = 0

    def record(self, description, matrix, note=""):
        has_fractions = any(v.denominator != 1 for row in matrix for v in row)
        copied = [row[:] for row in matrix]
        self.steps.append(Step(description, copied, note, has_fractions))

    @property
    def fraction_steps(self):
        return sum(1 for s in self.steps if s.has_fractions)

    @property
    def worst_denominator(self):
        worst = 1
        for step in self.steps:
            for row in step.matrix:
                for v in row:
                    worst = max(worst, v.denominator)
        return worst

    @property
    def largest_entry(self):
        biggest = 0
        for step in self.steps:
            for row in step.matrix:
                for v in row:
                    biggest = max(biggest, abs(v.numerator), abs(v.denominator))
        return biggest


# -------------------------------------------------------- row operations


def row_gcd(row):
    """
    Largest integer g such that dividing the row by g leaves every entry a
    whole number. Returns 1 when there is nothing worth cancelling.
    """
    nonzero = [v for v in row if v != 0]
    if not nonzero:
        return Fraction(1)
    if any(v.denominator != 1 for v in nonzero):
        return Fraction(1)
    g = 0
    for v in nonzero:
        g = gcd(g, abs(v.numerator))
    return Fraction(g) if g > 1 else Fraction(1)


def scale_row(row, factor):
    return [v * factor for v in row]


def combine_rows(target, source, target_mult, source_mult):
    """Return target_mult * target + source_mult * source."""
    return [target_mult * t + source_mult * s for t, s in zip(target, source)]


def elimination_multipliers(entry, pivot, human):
    """
    Work out the pair of multipliers that clears `entry` against `pivot`.

    Machine mode always divides: R_i -> R_i - (a/p) R_p.

    Human mode divides only when it comes out whole. Otherwise it scales
    the target row instead, R_i -> (p/g) R_i - (a/g) R_p, which clears the
    entry without ever introducing a denominator.
    """
    if not human:
        return Fraction(1), -entry / pivot

    ratio = entry / pivot
    if ratio.denominator == 1:
        return Fraction(1), -ratio

    g = gcd(
        entry.numerator * pivot.denominator,
        pivot.numerator * entry.denominator,
    )
    if g == 0:
        return pivot, -entry
    scale = Fraction(g, entry.denominator * pivot.denominator)
    return pivot / scale, -entry / scale


def canonicalize_contradiction(M, trace, is_augmented):
    """
    Finish off an inconsistent system.

    When a row reduces to all zero coefficients against a nonzero constant,
    that row is the proof the system has no solution. Neither strategy
    pivots in the constants column, so left alone that row keeps whatever
    scale it happened to land on, and the constants above it never get
    cleared. Two different routes would then stop at two different-looking
    matrices that both mean the same thing.

    So normalise it to read 0 = 1 and clear every other constant against it.
    That is what the true RREF of the augmented matrix looks like, and it is
    how the contradiction is written out by hand.
    """
    if not is_augmented or not M:
        return

    limit = len(M[0]) - 1
    target = None
    for r in range(len(M)):
        if all(M[r][c] == 0 for c in range(limit)) and M[r][-1] != 0:
            target = r
            break
    if target is None:
        return

    constant = M[target][-1]
    if constant != 1:
        inv = Fraction(1) / constant
        M[target] = scale_row(M[target], inv)
        trace.scalings += 1
        trace.record(
            "R%d -> %sR%d" % (target + 1, fmt_coeff(inv), target + 1),
            M,
            note="scale the contradiction row so it reads 0 = 1",
        )

    for i in range(len(M)):
        if i == target or M[i][-1] == 0:
            continue
        factor = M[i][-1]
        M[i] = combine_rows(M[i], M[target], Fraction(1), -factor)
        trace.eliminations += 1
        trace.record(
            describe_combination(i, target, Fraction(1), -factor),
            M,
            note="clear the constant against the contradiction row",
        )


def sink_zero_rows(M, trace):
    """
    Push rows that came out entirely zero to the bottom, which is where RREF
    wants them.

    Elimination can leave an empty row sitting above a row that still says
    something, and which row empties out first depends on the route taken.
    Sorting them down keeps the final answer independent of how it was
    reached. Every pivot row already sits above this region, so nothing that
    carries a leading 1 moves.
    """
    nonzero = [row for row in M if any(v != 0 for v in row)]
    zeros = [row for row in M if all(v == 0 for v in row)]
    if not zeros:
        return
    reordered = nonzero + zeros
    if reordered == M:
        return
    M[:] = reordered
    trace.swaps += 1
    trace.record("move empty rows to the bottom", M, note="RREF keeps empty rows last")


def describe_combination(target_idx, source_idx, target_mult, source_mult):
    """Write a row operation the way it appears in a textbook margin."""
    left = fmt_coeff(target_mult) + "R" + str(target_idx + 1)
    if source_mult < 0:
        joiner = "-"
        mag = -source_mult
    else:
        joiner = "+"
        mag = source_mult
    right = fmt_coeff(mag) + "R" + str(source_idx + 1)
    return "R%d -> %s %s %s" % (target_idx + 1, left, joiner, right)


# ------------------------------------------------------- pivot selection


def machine_pivot(M, start_row, col):
    """
    Partial pivoting: the largest magnitude entry in the column. This is what
    you do in floating point to keep rounding error small.
    """
    best_row = None
    best_mag = Fraction(0)
    for r in range(start_row, len(M)):
        mag = abs(M[r][col])
        if mag > best_mag:
            best_row = r
            best_mag = mag
    return best_row


def human_pivot(M, start_row, col):
    """
    Pick the pivot a person would pick, in order of preference:

      1. An entry of exactly 1. Eliminating with it never makes a fraction.
      2. An entry of -1. Just as clean, one sign to keep track of.
      3. An entry that divides every other entry in the column, so all the
         multipliers come out whole.
      4. Failing all that, the smallest entry, breaking ties toward the row
         holding the most zeros because that is the least arithmetic.
    """
    candidates = [r for r in range(start_row, len(M)) if M[r][col] != 0]
    if not candidates:
        return None

    for r in candidates:
        if M[r][col] == 1:
            return r
    for r in candidates:
        if M[r][col] == -1:
            return r

    for r in candidates:
        p = M[r][col]
        if p.denominator == 1:
            divides_all = True
            for o in candidates:
                if o != r and (M[o][col] / p).denominator != 1:
                    divides_all = False
                    break
            if divides_all:
                return r

    def sort_key(r):
        zeros = sum(1 for v in M[r] if v == 0)
        return (abs(M[r][col]), -zeros, M[r][col] < 0)

    return min(candidates, key=sort_key)


def find_manufactured_one(M, start_row, col):
    """
    The cheat code. When no row has a 1 in the pivot column, two rows whose
    entries differ by exactly 1 can make one: subtract and a leading 1 falls
    out for free.

    With a 3 and a 2 sitting in the column, R_i -> R_i - R_j turns the 3
    into a 1, and the whole column then eliminates without a single
    fraction. Costs one extra step, saves a page of arithmetic.
    """
    candidates = [r for r in range(start_row, len(M)) if M[r][col] != 0]
    if len(candidates) < 2:
        return None
    if any(abs(M[r][col]) == 1 for r in candidates):
        return None

    fallback = None
    for i in candidates:
        for j in candidates:
            if i == j:
                continue
            diff = M[i][col] - M[j][col]
            if diff == 1:
                return (i, j)
            if diff == -1 and fallback is None:
                fallback = (i, j)
    return fallback


# -------------------------------------------------------------- reduction


def reduce_matrix(aug_matrix, is_augmented=True, mode="human"):
    """
    Row reduce to REF then RREF using exact rational arithmetic.

    Returns (REF, RREF, pivot_positions, trace).
    """
    M = [[Fraction(v) for v in row] for row in aug_matrix]
    num_rows = len(M)
    num_cols = len(M[0])
    col_limit = num_cols - 1 if is_augmented else num_cols
    human = mode == "human"

    trace = Trace()
    trace.record("Initial matrix", M)

    pivots = []
    row = 0
    col = 0

    # ---- forward phase
    while row < num_rows and col < col_limit:
        if human:
            manufactured = find_manufactured_one(M, row, col)
            if manufactured is not None:
                i, j = manufactured
                M[i] = combine_rows(M[i], M[j], Fraction(1), Fraction(-1))
                trace.eliminations += 1
                trace.record(
                    describe_combination(i, j, Fraction(1), Fraction(-1)),
                    M,
                    note="make a 1 in the pivot column so nothing below needs dividing",
                )

        pivot_row = human_pivot(M, row, col) if human else machine_pivot(M, row, col)
        if pivot_row is None:
            col += 1
            continue

        if pivot_row != row:
            M[row], M[pivot_row] = M[pivot_row], M[row]
            trace.swaps += 1
            note = "bring the friendliest pivot up" if human else "partial pivoting"
            trace.record("R%d <-> R%d" % (row + 1, pivot_row + 1), M, note=note)

        pivot = M[row][col]

        # Machine mode normalises straight away. Human mode refuses to
        # divide here, because that is exactly where fractions come from.
        if not human and pivot != 1:
            inv = Fraction(1) / pivot
            M[row] = scale_row(M[row], inv)
            trace.scalings += 1
            trace.record(
                "R%d -> %sR%d" % (row + 1, fmt_coeff(inv), row + 1),
                M,
                note="scale the pivot to 1",
            )
            pivot = M[row][col]

        for r in range(row + 1, num_rows):
            entry = M[r][col]
            if entry == 0:
                continue
            t_mult, s_mult = elimination_multipliers(entry, pivot, human)
            M[r] = combine_rows(M[r], M[row], t_mult, s_mult)
            trace.eliminations += 1
            trace.record(describe_combination(r, row, t_mult, s_mult), M)

            if human:
                g = row_gcd(M[r])
                if g != 1:
                    M[r] = scale_row(M[r], Fraction(1) / g)
                    trace.scalings += 1
                    trace.record(
                        "R%d -> %sR%d" % (r + 1, fmt_coeff(Fraction(1) / g), r + 1),
                        M,
                        note="every entry divides by %s, so cancel it out" % fmt(g),
                    )

        pivots.append((row, col))
        row += 1
        col += 1

    REF = [r[:] for r in M]

    # ---- backward phase
    for r, c in reversed(pivots):
        pivot = M[r][c]
        for i in range(r):
            entry = M[i][c]
            if entry == 0:
                continue
            t_mult, s_mult = elimination_multipliers(entry, pivot, human)
            M[i] = combine_rows(M[i], M[r], t_mult, s_mult)
            trace.eliminations += 1
            trace.record(
                describe_combination(i, r, t_mult, s_mult),
                M,
                note="clear above the pivot",
            )

            if human:
                g = row_gcd(M[i])
                if g != 1:
                    M[i] = scale_row(M[i], Fraction(1) / g)
                    trace.scalings += 1
                    trace.record(
                        "R%d -> %sR%d" % (i + 1, fmt_coeff(Fraction(1) / g), i + 1),
                        M,
                        note="every entry divides by %s, so cancel it out" % fmt(g),
                    )

    # Human mode saved every division for last. Do them now, once each.
    if human:
        for r, c in pivots:
            pivot = M[r][c]
            if pivot != 1:
                inv = Fraction(1) / pivot
                M[r] = scale_row(M[r], inv)
                trace.scalings += 1
                trace.record(
                    "R%d -> %sR%d" % (r + 1, fmt_coeff(inv), r + 1),
                    M,
                    note="only now divide, once, to land the leading 1",
                )

    canonicalize_contradiction(M, trace, is_augmented)
    sink_zero_rows(M, trace)

    return REF, M, pivots, trace


# --------------------------------------------------------------- analysis


def solve_system(M, pivots):
    """
    Read the solution off a reduced augmented matrix.

    Returns ("INCONSISTENT", None), ("UNIQUE", {col: value}), or
    ("INFINITE", (leading_expressions, free_variable_names)).
    """
    num_cols = len(M[0])
    num_vars = num_cols - 1

    for row in M:
        if all(v == 0 for v in row[:-1]) and row[-1] != 0:
            return "INCONSISTENT", None

    lead_cols = set(c for _, c in pivots)
    free_cols = sorted(set(range(num_vars)) - lead_cols)

    if not free_cols:
        return "UNIQUE", dict((c, M[r][-1]) for r, c in pivots)

    params = {}
    for i, fc in enumerate(free_cols):
        params[fc] = "t%d" % (i + 1)

    leading = {}
    for r, c in pivots:
        terms = []
        constant = M[r][-1]
        if constant != 0:
            terms.append(fmt(constant))
        for fc in free_cols:
            coeff = -M[r][fc]
            if coeff == 0:
                continue
            mag = abs(coeff)
            body = params[fc] if mag == 1 else fmt(mag) + params[fc]
            if not terms:
                terms.append(body if coeff > 0 else "-" + body)
            else:
                terms.append((" + " if coeff > 0 else " - ") + body)
        leading[c] = "".join(terms) if terms else "0"

    return "INFINITE", (leading, params)


def print_steps(trace, is_augmented):
    for i, step in enumerate(trace.steps):
        header = "Step %d: %s" % (i, step.description) if i else step.description
        print(Colors.BLUE + Colors.BOLD + header + Colors.ENDC)
        if step.note:
            print(Colors.DIM + "  (" + step.note + ")" + Colors.ENDC)
        print(format_matrix(step.matrix, is_augmented))
        print()


def print_report(matrix, is_augmented, mode, show_steps=True):
    REF, RREF, pivots, trace = reduce_matrix(matrix, is_augmented, mode)

    label = "HUMAN" if mode == "human" else "MACHINE"
    banner = "%s STRATEGY" % label
    print(Colors.HEADER + Colors.BOLD + "=" * 56 + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + banner + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + "=" * 56 + Colors.ENDC)
    print()

    if show_steps:
        print_steps(trace, is_augmented)

    print(Colors.BOLD + "Row-Echelon Form (REF):" + Colors.ENDC)
    print(format_matrix(REF, is_augmented))
    print()
    print(Colors.BOLD + "Reduced Row-Echelon Form (RREF):" + Colors.ENDC)
    print(format_matrix(RREF, is_augmented))
    print()

    num_rows = len(matrix)
    num_cols = len(matrix[0])
    num_vars = num_cols - 1 if is_augmented else num_cols
    rank = len(pivots)

    print(Colors.BOLD + "Matrix properties:" + Colors.ENDC)
    print("  Rank: %s%d%s" % (Colors.BLUE, rank, Colors.ENDC))
    if is_augmented:
        print("  Nullity: %s%d%s" % (Colors.BLUE, num_vars - rank, Colors.ENDC))
        print("  Variables: %d" % num_vars)
        print("  Equations: %d" % num_rows)
    else:
        print("  Dimensions: %d x %d" % (num_rows, num_cols))
        if num_rows == num_cols:
            invertible = rank == num_rows
            color = Colors.GREEN if invertible else Colors.FAIL
            print("  Invertible: %s%s%s" % (color, invertible, Colors.ENDC))
    print()

    print(Colors.BOLD + "Work done:" + Colors.ENDC)
    print("  Row operations: %d" % (len(trace.steps) - 1))
    print("  Swaps: %d   Scalings: %d   Eliminations: %d"
          % (trace.swaps, trace.scalings, trace.eliminations))
    print("  Steps containing a fraction: %s%d of %d%s"
          % (Colors.BLUE, trace.fraction_steps, len(trace.steps), Colors.ENDC))
    print("  Worst denominator seen: %d" % trace.worst_denominator)
    print("  Largest number seen: %d" % trace.largest_entry)
    print()

    if is_augmented:
        status, payload = solve_system(RREF, pivots)
        if status == "INCONSISTENT":
            print(Colors.FAIL + Colors.BOLD
                  + "Conclusion: inconsistent, no solution." + Colors.ENDC)
            print("A row reduced to [ 0 ... 0 | nonzero ], which says 0 = 1.")
        elif status == "UNIQUE":
            print(Colors.GREEN + Colors.BOLD
                  + "Conclusion: consistent, one unique solution." + Colors.ENDC)
            for c in sorted(payload):
                print("  x%d = %s%s%s"
                      % (c + 1, Colors.BOLD, fmt(payload[c]), Colors.ENDC))
        else:
            leading, params = payload
            print(Colors.GREEN + Colors.BOLD
                  + "Conclusion: consistent, infinitely many solutions."
                  + Colors.ENDC)
            print("Free variables: " + ", ".join(
                "x%d = %s" % (c + 1, params[c]) for c in sorted(params)))
            print()
            print("General solution:")
            for idx in range(num_vars):
                value = leading.get(idx, params.get(idx))
                print("  x%d = %s%s%s"
                      % (idx + 1, Colors.BOLD, value, Colors.ENDC))
    print()
    return trace


def print_comparison(matrix, is_augmented, show_steps=False):
    """Run both strategies on the same matrix and contrast the scratch work."""
    human_trace = print_report(matrix, is_augmented, "human", show_steps)
    machine_trace = print_report(matrix, is_augmented, "machine", show_steps)

    print(Colors.HEADER + Colors.BOLD + "=" * 56 + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + "SIDE BY SIDE" + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + "=" * 56 + Colors.ENDC)
    print()
    rows = [
        ("Row operations",
         len(human_trace.steps) - 1, len(machine_trace.steps) - 1),
        ("Steps with a fraction",
         human_trace.fraction_steps, machine_trace.fraction_steps),
        ("Worst denominator",
         human_trace.worst_denominator, machine_trace.worst_denominator),
        ("Largest number",
         human_trace.largest_entry, machine_trace.largest_entry),
    ]
    print("  %-24s %10s %10s" % ("", "human", "machine"))
    for label, h, m in rows:
        print("  %-24s %10s %10s" % (label, h, m))
    print()
    if human_trace.fraction_steps < machine_trace.fraction_steps:
        print("The human route stayed on whole numbers longer, which is the "
              "whole point of it.")
    elif len(human_trace.steps) > len(machine_trace.steps):
        print("The machine route was shorter here. The human route traded "
              "extra steps for easier arithmetic.")
    else:
        print("Both routes cost about the same on this matrix.")
    print()


# ------------------------------------------------------------ sample set


def sample_problems():
    return {
        "1": {
            "title": "3x3, inconsistent",
            "matrix": [[2, -1, 3, -1], [1, 1, -2, 1], [4, 1, -1, 3]],
            "is_augmented": True,
        },
        "2": {
            "title": "3x3, sparse with a natural 1",
            "matrix": [[-8, 0, 1, 9], [0, 1, -1, 0], [1, 8, 0, 0]],
            "is_augmented": True,
        },
        "3": {
            "title": "4 equations, 3 unknowns, consistent",
            "matrix": [[1, 1, -1, -2], [1, 1, 1, 0], [1, 0, -1, 1], [0, 1, -4, -7]],
            "is_augmented": True,
        },
        "4": {
            "title": "3x3, inconsistent with a repeated row pattern",
            "matrix": [[-1, 2, -1, 1], [2, 8, 1, -2], [1, -2, 1, 1]],
            "is_augmented": True,
        },
        "5": {
            "title": "3x3, unique solution",
            "matrix": [[1, -2, 4, 0], [-1, 1, -2, -1], [1, 5, 1, 2]],
            "is_augmented": True,
        },
        "6": {
            "title": "4x4, unique solution",
            "matrix": [
                [1, -1, 1, 1, 0],
                [2, 1, -1, 1, -1],
                [3, -4, -1, 1, 1],
                [-1, 1, 1, -1, 1],
            ],
            "is_augmented": True,
        },
        "7": {
            "title": "4x4, inconsistent, no easy pivots",
            "matrix": [
                [2, 1, -1, 3, 10],
                [-3, -1, 2, 2, 9],
                [8, 2, 1, 1, 0],
                [4, 1, 4, 8, -1],
            ],
            "is_augmented": True,
        },
        "8": {
            "title": "3x4, infinitely many solutions",
            "matrix": [[1, 2, 3, 4, 5], [2, 4, 8, 10, 14], [3, 6, 11, 14, 19]],
            "is_augmented": True,
        },
        "9": {
            "title": "ugly pivots, where the human route pays off",
            "matrix": [[6, 9, 15, 3], [4, 7, 11, 5], [8, 13, 21, 9]],
            "is_augmented": True,
        },
    }


# ------------------------------------------------------------------- input


def parse_value(text):
    text = text.strip()
    if "/" in text:
        num, denom = text.split("/", 1)
        return Fraction(int(num), int(denom))
    if "." in text or "e" in text.lower():
        return Fraction(str(text)).limit_denominator(10 ** 9)
    return Fraction(int(text))


def parse_matrix_text(text):
    """Parse rows separated by ';' or newlines, values by spaces or commas."""
    rows = []
    chunks = [c for c in text.replace("\n", ";").split(";") if c.strip()]
    for chunk in chunks:
        cells = chunk.replace(",", " ").split()
        rows.append([parse_value(c) for c in cells])
    if not rows:
        raise ValueError("no rows found")
    width = len(rows[0])
    for r in rows:
        if len(r) != width:
            raise ValueError("all rows must have the same number of entries")
    return rows


# -------------------------------------------------------------- interactive


def choose_mode():
    print()
    print(Colors.BOLD + "How should it solve?" + Colors.ENDC)
    print("  1. Human   - avoid fractions, keep the arithmetic easy")
    print("  2. Machine - straight partial pivoting, fewest steps")
    print("  3. Both    - solve twice and compare")
    choice = input("Choose (1-3, default 1): ").strip()
    if choice == "2":
        return "machine"
    if choice == "3":
        return "both"
    return "human"


def run_choice(matrix, is_augmented, mode, show_steps):
    if mode == "both":
        print_comparison(matrix, is_augmented, show_steps)
    else:
        print_report(matrix, is_augmented, mode, show_steps)


def sample_menu():
    problems = sample_problems()
    while True:
        print()
        print(Colors.HEADER + Colors.BOLD + "=== SAMPLE PROBLEMS ===" + Colors.ENDC)
        for key in sorted(problems):
            print("  %s. %s" % (key, problems[key]["title"]))
        print("  0. Back")
        choice = input("\nPick a problem: ").strip()
        if choice == "0":
            return
        if choice not in problems:
            print(Colors.FAIL + "No such problem." + Colors.ENDC)
            continue
        problem = problems[choice]
        mode = choose_mode()
        print("\nSolving: " + Colors.BOLD + problem["title"] + Colors.ENDC + "\n")
        run_choice(problem["matrix"], problem["is_augmented"], mode, True)
        input("Press Enter to continue...")


def custom_menu():
    print()
    print(Colors.HEADER + Colors.BOLD + "=== CUSTOM MATRIX ===" + Colors.ENDC)
    print("Enter one row per line. Integers, fractions like -2/3, or decimals.")
    print("Finish with a blank line.")
    print()
    lines = []
    while True:
        try:
            line = input("Row %d: " % (len(lines) + 1)).strip()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)

    if not lines:
        print(Colors.WARNING + "Nothing entered." + Colors.ENDC)
        return

    try:
        matrix = parse_matrix_text(";".join(lines))
    except Exception as exc:
        print(Colors.FAIL + "Could not read that: %s" % exc + Colors.ENDC)
        return

    augmented = input("Is the last column a constants column? (Y/n): ").strip().lower()
    is_augmented = augmented != "n"
    mode = choose_mode()
    steps = input("Show every step? (Y/n): ").strip().lower() != "n"
    print()
    run_choice(matrix, is_augmented, mode, steps)
    input("Press Enter to continue...")


def interactive():
    while True:
        print()
        print(Colors.HEADER + Colors.BOLD
              + "=== GAUSSIAN ELIMINATION SOLVER ===" + Colors.ENDC)
        print("  1. Solve a sample problem")
        print("  2. Enter your own matrix")
        print("  3. Exit")
        choice = input("\nChoose (1-3): ").strip()
        if choice == "1":
            sample_menu()
        elif choice == "2":
            custom_menu()
        elif choice == "3":
            print()
            return
        else:
            print(Colors.FAIL + "Not an option." + Colors.ENDC)


# -------------------------------------------------------------------- cli


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gaussian_solver",
        description="Row reduce a matrix to RREF, the human way or the machine way.",
        epilog=(
            "Example: gaussian_solver.py -m \"2 -1 3 -1; 1 1 -2 1; 4 1 -1 3\" "
            "--mode both"
        ),
    )
    parser.add_argument(
        "-m", "--matrix",
        help="matrix as rows separated by ';', e.g. \"1 2 3; 4 5 6\"",
    )
    parser.add_argument(
        "--mode", choices=["human", "machine", "both"], default="human",
        help="strategy to use (default: human)",
    )
    parser.add_argument(
        "--no-augmented", action="store_true",
        help="treat the last column as a normal column, not constants",
    )
    parser.add_argument(
        "--no-steps", action="store_true",
        help="print only the result, not every row operation",
    )
    parser.add_argument(
        "--sample", help="run one of the built-in sample problems by number",
    )
    parser.add_argument(
        "--list-samples", action="store_true", help="list the sample problems",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="disable ANSI colors",
    )
    args = parser.parse_args(argv)

    if args.no_color:
        disable_colors()

    if args.list_samples:
        problems = sample_problems()
        for key in sorted(problems):
            print("%s. %s" % (key, problems[key]["title"]))
        return 0

    if args.sample:
        problems = sample_problems()
        if args.sample not in problems:
            print("No sample numbered %s. Try --list-samples." % args.sample)
            return 1
        problem = problems[args.sample]
        print("Solving: " + problem["title"] + "\n")
        run_choice(problem["matrix"], problem["is_augmented"],
                   args.mode, not args.no_steps)
        return 0

    if args.matrix:
        try:
            matrix = parse_matrix_text(args.matrix)
        except Exception as exc:
            print("Could not read that matrix: %s" % exc)
            return 1
        run_choice(matrix, not args.no_augmented, args.mode, not args.no_steps)
        return 0

    interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
