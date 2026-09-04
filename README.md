# Gaussian Solver

Row reduces a matrix to reduced row echelon form and shows every step. It
solves the same problem two different ways, and the interesting part is the
difference between them.

Exact rational arithmetic throughout, so `1/3` stays `1/3` and never drifts
into `0.33333333`. No dependencies, standard library only.

## The two strategies

**Machine.** Partial pivoting, the way a numerical library does it. Take the
largest entry in the column, scale it to 1, eliminate everything below. It
makes the fewest decisions and it is what you want in floating point, where
dividing by the largest available number keeps rounding error small. Working
exactly, though, that first division drops a fraction into the matrix on step
one and every later step drags it along.

**Human.** The way a person does it on paper, where the enemy is not rounding
error but arithmetic you have to do in your head.

- **Hunt for a pivot of 1 or -1 first.** Eliminating with a pivot of 1 never
  creates a fraction. This one rule does most of the work.
- **Refuse to divide.** Machine mode scales the pivot row to 1 immediately.
  Human mode leaves it alone, because that division is where the fractions
  come from. It carries whole numbers through the entire elimination and
  divides once at the very end to land the leading 1.
- **Scale instead of divide.** With no clean pivot available, rather than
  `R2 -> R2 - (4/6)R1`, do `R2 -> 3R2 - 2R1`. Same zero in the same place,
  no denominator.
- **Cancel common factors.** The moment every entry in a row shares a factor,
  divide it out. A row of `[0, -3, -3, -9]` becomes `[0, -1, -1, -3]` and
  everything downstream gets easier.
- **Manufacture a 1 when none exists.** If two rows have entries in the pivot
  column that differ by exactly 1, subtracting them produces a 1 for free.
  A 3 and a 2 in the column become a 1 in one step, and the column then
  clears without a single fraction. One extra step to save a page of work.
- **Prefer sparse rows.** Ties break toward the row with the most zeros,
  because that is the least arithmetic.

Both strategies produce the same RREF. That is checked against 3000 randomized
matrices, which also confirms the human route hits a fraction about a quarter
as often.

## Example

The two routes on the same system, same number of row operations:

```
                                human    machine
  Row operations                   11         11
  Steps with a fraction             0          8
  Worst denominator                 1          8
```

Same answer, same effort, and one of them you could have done on paper.

## Usage

Needs Python 3.7 or newer. Nothing to install.

```bash
# solve a matrix, rows separated by semicolons
python gaussian_solver.py -m "2 -1 3 -1; 1 1 -2 1; 4 1 -1 3"

# compare the two strategies side by side
python gaussian_solver.py -m "6 9 15 3; 4 7 11 5; 8 13 21 9" --mode both

# just the answer, no step-by-step
python gaussian_solver.py -m "1 2 3; 4 5 6" --no-steps

# built-in problems
python gaussian_solver.py --list-samples
python gaussian_solver.py --sample 9 --mode human

# menu-driven, if you would rather be asked
python gaussian_solver.py
```

### Options

| Flag | Meaning |
|---|---|
| `-m`, `--matrix` | Matrix as rows separated by `;`, values by spaces or commas |
| `--mode` | `human`, `machine`, or `both` (default `human`) |
| `--no-augmented` | Treat the last column as a normal column, not constants |
| `--no-steps` | Print only the result |
| `--sample N` | Run built-in problem N |
| `--list-samples` | List the built-in problems |
| `--no-color` | Disable ANSI color |

Entries can be integers, fractions like `-2/3`, or decimals like `1.5`.
By default the last column is treated as the constants of a linear system;
pass `--no-augmented` to reduce a plain matrix instead.

## What it reports

- Every row operation in textbook notation, `R2 -> 2R2 - 3R1`, with the
  matrix after each one and a note explaining any non-obvious choice
- Row echelon form and reduced row echelon form
- Rank, nullity, and invertibility for a square matrix
- A count of how much arithmetic the route cost, including how many steps
  contained a fraction and the worst denominator reached
- The solution: unique, inconsistent, or a general parametric form with the
  free variables named

For an inconsistent system the contradiction row is normalized to read
`0 = 1` and the constants above it are cleared, which is both the true RREF
of the augmented matrix and how you would write out the contradiction by
hand.

## License

MIT
