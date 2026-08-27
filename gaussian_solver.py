import sys
import os
from fractions import Fraction

# ANSI terminal colors (fallback to empty strings if not a terminal or not supported)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Disable colors if we are not outputting to a TTY (e.g. redirected to a file or in some runtimes)
if not sys.stdout.isatty():
    for attr in dir(Colors):
        if not attr.startswith('__'):
            setattr(Colors, attr, '')

def format_fraction(val):
    """Formats a Fraction object to a clean string representation."""
    if val.denominator == 1:
        return f"{val.numerator}"
    return f"{val.numerator}/{val.denominator}"

def format_matrix(matrix, is_augmented=True):
    """Formats a matrix with vertical bar for augmented column if applicable."""
    lines = []
    # Find max width for each column to align them beautifully
    col_widths = []
    num_cols = len(matrix[0])
    for c in range(num_cols):
        max_w = 0
        for r in range(len(matrix)):
            w = len(format_fraction(matrix[r][c]))
            if w > max_w:
                max_w = w
        col_widths.append(max_w)
        
    for row in matrix:
        row_strs = []
        for c, val in enumerate(row):
            val_str = format_fraction(val)
            # Right-align numbers
            aligned = f"{val_str:>{col_widths[c]}}"
            row_strs.append(aligned)
        
        if is_augmented and num_cols > 1:
            # Join everything except the last column, then add ' | ', then add last column
            left_part = "  ".join(row_strs[:-1])
            right_part = row_strs[-1]
            lines.append(f"[  {left_part}  │  {right_part}  ]")
        else:
            lines.append("[  " + "  ".join(row_strs) + "  ]")
    return "\n".join(lines)

def print_step(matrix, step_num, operation_desc, is_augmented=True):
    """Prints a row reduction step with operation description."""
    header = f"Step {step_num}: {operation_desc}"
    print(f"{Colors.BLUE}{Colors.BOLD}{header}{Colors.ENDC}")
    print(f"{Colors.GREEN}{'-' * len(header)}{Colors.ENDC}")
    print(format_matrix(matrix, is_augmented))
    print()

def solve_system_parametric(M, leading_positions, is_augmented=True):
    """
    Analyzes the row-reduced augmented matrix M and returns consistency,
    unique solution, or parametric general solutions.
    """
    num_rows = len(M)
    num_cols = len(M[0])
    
    # If not augmented, we just describe RREF, no system solving
    if not is_augmented:
        return None
        
    # 1. Consistency Check: Look for a row of [0, 0, ..., 0 | non-zero]
    inconsistent = False
    for r in range(num_rows):
        if all(val == 0 for val in M[r][:-1]) and M[r][-1] != 0:
            inconsistent = True
            break
            
    if inconsistent:
        return "INCONSISTENT", None
        
    # 2. Identify variables
    num_vars = num_cols - 1
    all_vars = set(range(num_vars))
    lead_vars = set(lead for r, lead in leading_positions)
    free_vars = sorted(list(all_vars - lead_vars))
    
    if len(free_vars) == 0:
        # Unique solution
        sol = {}
        for r, lead in leading_positions:
            sol[lead] = M[r][-1]
        return "UNIQUE", sol
    else:
        # Infinitely many solutions (parametric)
        params = {fv: f"t_{i+1}" for i, fv in enumerate(free_vars)}
        sol_leading = {}
        for r, lead in leading_positions:
            constant = M[r][-1]
            terms = []
            if constant != 0:
                terms.append(f"{format_fraction(constant)}")
                
            for fv in free_vars:
                coeff = -M[r][fv] # x_lead = constant - coeff*x_fv
                if coeff != 0:
                    sign = "+" if coeff > 0 else "-"
                    abs_coeff = abs(coeff)
                    coeff_str = "" if abs_coeff == 1 else f"{format_fraction(abs_coeff)}*"
                    
                    if len(terms) == 0:
                        if coeff > 0:
                            terms.append(f"{coeff_str}{params[fv]}")
                        else:
                            terms.append(f"-{coeff_str}{params[fv]}")
                    else:
                        terms.append(f" {sign} {coeff_str}{params[fv]}")
            
            sol_leading[lead] = "".join(terms) if terms else "0"
            
        sol_free = {fv: params[fv] for fv in free_vars}
        return "INFINITE", (sol_leading, sol_free)

def reduce_matrix(aug_matrix, is_augmented=True, show_steps=True):
    """
    Performs full row-reduction (REF and RREF) using exact Fraction arithmetic.
    """
    M = [[Fraction(val) for val in row] for row in aug_matrix]
    num_rows = len(M)
    num_cols = len(M[0])
    col_limit = num_cols - 1 if is_augmented else num_cols
    
    step_num = 1
    if show_steps:
        print_step(M, step_num, "Initial Augmented Matrix" if is_augmented else "Initial Matrix", is_augmented)
        step_num += 1
        
    lead = 0
    leading_positions = [] # list of (row, col)
    
    # Forward Phase (REF)
    for r in range(num_rows):
        if lead >= col_limit:
            break
            
        # Find a non-zero pivot entry in column 'lead' at or below row 'r'
        pivot_row = r
        while M[pivot_row][lead] == 0:
            pivot_row += 1
            if pivot_row == num_rows:
                pivot_row = r
                lead += 1
                if lead == col_limit:
                    break
        else:
            # Swap rows if necessary
            if pivot_row != r:
                M[r], M[pivot_row] = M[pivot_row], M[r]
                if show_steps:
                    print_step(M, step_num, f"Swap Row {r+1} and Row {pivot_row+1}", is_augmented)
                    step_num += 1
            
            # Make pivot 1 (leading 1)
            pivot_val = M[r][lead]
            if pivot_val != 1:
                M[r] = [val / pivot_val for val in M[r]]
                if show_steps:
                    print_step(M, step_num, f"Multiply Row {r+1} by {format_fraction(Fraction(1, pivot_val))}", is_augmented)
                    step_num += 1
            
            # Eliminate entries below pivot row
            for i in range(r + 1, num_rows):
                factor = M[i][lead]
                if factor != 0:
                    M[i] = [val_i - factor * val_r for val_i, val_r in zip(M[i], M[r])]
                    if show_steps:
                        print_step(M, step_num, f"Add {format_fraction(-factor)} * Row {r+1} to Row {i+1}", is_augmented)
                        step_num += 1
            
            leading_positions.append((r, lead))
            lead += 1
            continue
        break
        
    # Store Row Echelon Form (REF)
    REF_matrix = [[val for val in row] for row in M]
    
    # Backward Phase (RREF / Gauss-Jordan)
    for r, lead in reversed(leading_positions):
        for i in range(r):
            factor = M[i][lead]
            if factor != 0:
                M[i] = [val_i - factor * val_r for val_i, val_r in zip(M[i], M[r])]
                if show_steps:
                    print_step(M, step_num, f"Add {format_fraction(-factor)} * Row {r+1} to Row {i+1} (eliminate upwards)", is_augmented)
                    step_num += 1
                    
    RREF_matrix = M
    return REF_matrix, RREF_matrix, leading_positions

def run_solver(matrix, is_augmented=True, show_steps=True):
    """Helper to run reduction and display solution conclusions."""
    REF, RREF, pivots = reduce_matrix(matrix, is_augmented, show_steps)
    
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 50}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}ELIMINATION COMPLETE{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 50}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Row-Echelon Form (REF):{Colors.ENDC}")
    print(format_matrix(REF, is_augmented))
    print()
    
    print(f"{Colors.BOLD}Reduced Row-Echelon Form (RREF):{Colors.ENDC}")
    print(format_matrix(RREF, is_augmented))
    print()
    
    # Calculate dimensions
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    num_vars = num_cols - 1 if is_augmented else num_cols
    
    rank = len(pivots)
    nullity = num_vars - rank
    
    print(f"{Colors.BOLD}Matrix Properties:{Colors.ENDC}")
    print(f"  • Rank (number of leading 1s): {Colors.BLUE}{rank}{Colors.ENDC}")
    if is_augmented:
        print(f"  • Nullity (dimension of null space): {Colors.BLUE}{nullity}{Colors.ENDC}")
        print(f"  • Number of variables: {num_vars}")
        print(f"  • Number of equations: {num_rows}")
    else:
        print(f"  • Dimensions: {num_rows} x {num_cols}")
        if num_rows == num_cols:
            is_inv = (rank == num_rows)
            print(f"  • Invertible: {Colors.GREEN if is_inv else Colors.FAIL}{is_inv}{Colors.ENDC}")
            
    print()
    
    if is_augmented:
        sol_status = solve_system_parametric(RREF, pivots, is_augmented)
        if sol_status[0] == "INCONSISTENT":
            print(f"{Colors.FAIL}{Colors.BOLD}Conclusion: The system is INCONSISTENT (no solution).{Colors.ENDC}")
            print(f"Explanation: A row-echelon row of the form [ 0 0 ... 0 | 1 ] was found, implying 0 = 1.")
        elif sol_status[0] == "UNIQUE":
            sol = sol_status[1]
            print(f"{Colors.GREEN}{Colors.BOLD}Conclusion: The system is CONSISTENT with a UNIQUE solution:{Colors.ENDC}")
            for k, v in sorted(sol.items()):
                print(f"  x_{k+1} = {Colors.BOLD}{format_fraction(v)}{Colors.ENDC}")
        else:
            sol_leading, sol_free = sol_status[1]
            print(f"{Colors.GREEN}{Colors.BOLD}Conclusion: The system is CONSISTENT with INFINITELY MANY solutions:{Colors.ENDC}")
            print("Parameters (free variables):")
            for k, v in sorted(sol_free.items()):
                print(f"  x_{k+1} = {v}")
            print("\nGeneral Parametric Solution:")
            for idx in range(num_vars):
                if idx in sol_leading:
                    print(f"  x_{idx+1} = {Colors.BOLD}{sol_leading[idx]}{Colors.ENDC}")
                else:
                    print(f"  x_{idx+1} = {Colors.BOLD}{sol_free[idx]}{Colors.ENDC}")
    print()

def get_textbook_problems():
    return {
        "1": {
            "title": "Systems of Eqns 1 - Problem 1 (Inconsistent)",
            "matrix": [
                [2, -1, 3, -1],
                [1, 1, -2, 1],
                [4, 1, -1, 3]
            ],
            "is_augmented": True
        },
        "2": {
            "title": "Systems of Eqns 1 - Problem 2 (REF Translation Matrix)",
            "matrix": [
                [-8, 0, 1, 9],
                [0, 1, -1, 0],
                [1, 8, 0, 0]
            ],
            "is_augmented": True
        },
        "3": {
            "title": "Systems of Eqns 1 - Problem 4 (Consistent, a = -4)",
            "matrix": [
                [1, 1, -1, -2],
                [1, 1, 1, 0],
                [1, 0, -1, 1],
                [0, 1, -4, -7]
            ],
            "is_augmented": True
        },
        "4": {
            "title": "Systems of Eqns 1 - Problem 5 (Inconsistent, a = 1)",
            "matrix": [
                [-1, 2, -1, 1],
                [2, 8, 1, -2],
                [1, -2, 1, 1]
            ],
            "is_augmented": True
        },
        "5": {
            "title": "Systems of Eqns 1 - Problem 6 (Gaussian Elimination to Unique Sol)",
            "matrix": [
                [1, -2, 4, 0],
                [-1, 1, -2, -1],
                [1, 5, 1, 2]
            ],
            "is_augmented": True
        },
        "6": {
            "title": "Systems of Eqns 1 - Problem 7 (4-Variable Gauss-Jordan to Unique Sol)",
            "matrix": [
                [1, -1, 1, 1, 0],
                [2, 1, -1, 1, -1],
                [3, -4, -1, 1, 1],
                [-1, 1, 1, -1, 1]
            ],
            "is_augmented": True
        },
        "7": {
            "title": "Systems of Eqns 1 - Problem 8 (4-Variable Inconsistent)",
            "matrix": [
                [2, 1, -1, 3, 10],
                [-3, -1, 2, 2, 9],
                [8, 2, 1, 1, 0],
                [4, 1, 4, 8, -1]
            ],
            "is_augmented": True
        }
    }

def show_textbook_menu():
    problems = get_textbook_problems()
    while True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== PRE-LOADED TEXTBOOK & PACKET PROBLEMS ==={Colors.ENDC}")
        for k, v in problems.items():
            print(f"  {k}. {v['title']}")
        print("  0. Back to Main Menu")
        
        choice = input("\nSelect a problem to solve (0-7): ").strip()
        if choice == "0":
            return
        elif choice in problems:
            prob = problems[choice]
            print(f"\nLoading: {Colors.BOLD}{prob['title']}{Colors.ENDC}")
            run_solver(prob['matrix'], prob['is_augmented'], show_steps=True)
            input("Press Enter to continue...")
        else:
            print(f"{Colors.FAIL}Invalid choice. Please try again.{Colors.ENDC}")

def input_custom_matrix():
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== DEFINE CUSTOM MATRIX ==={Colors.ENDC}")
    try:
        rows = int(input("Enter the number of rows: ").strip())
        cols = int(input("Enter the number of columns: ").strip())
        
        is_augmented_str = input("Is this an augmented system matrix? (y/n, default=y): ").strip().lower()
        is_augmented = is_augmented_str != 'n'
        
        print(f"\nEnter the elements row-by-row. You can enter integers (e.g. 5), fractions (e.g. -2/3), or decimals (e.g. 1.5).")
        matrix = []
        for r in range(rows):
            while True:
                row_input = input(f"Row {r+1} (separate {cols} values with spaces): ").strip().split()
                if len(row_input) != cols:
                    print(f"{Colors.WARNING}Error: You must enter exactly {cols} values. Please try again.{Colors.ENDC}")
                    continue
                try:
                    row_vals = []
                    for val in row_input:
                        if '/' in val:
                            num, denom = val.split('/')
                            row_vals.append(Fraction(int(num), int(denom)))
                        elif '.' in val:
                            row_vals.append(Fraction(float(val)))
                        else:
                            row_vals.append(Fraction(int(val)))
                    matrix.append(row_vals)
                    break
                except Exception as e:
                    print(f"{Colors.WARNING}Error parsing numbers: {e}. Please try again.{Colors.ENDC}")
                    
        show_steps_str = input("\nShow step-by-step row operations? (y/n, default=y): ").strip().lower()
        show_steps = show_steps_str != 'n'
        
        run_solver(matrix, is_augmented, show_steps)
        input("Press Enter to continue...")
    except Exception as e:
        print(f"{Colors.FAIL}An error occurred: {e}{Colors.ENDC}")
        input("Press Enter to return...")

def main():
    while True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== GAUSSIAN ELIMINATION SOLVER ==={Colors.ENDC}")
        print("  1. Solve a Pre-Loaded Class/Study Packet Problem")
        print("  2. Input and Solve a Custom Matrix System")
        print("  3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            show_textbook_menu()
        elif choice == "2":
            input_custom_matrix()
        elif choice == "3":
            print(f"\n{Colors.GREEN}Thank you for using the Gaussian Elimination Solver! Good luck!{Colors.ENDC}\n")
            break
        else:
            print(f"{Colors.FAIL}Invalid choice. Please try again.{Colors.ENDC}")

if __name__ == "__main__":
    main()
