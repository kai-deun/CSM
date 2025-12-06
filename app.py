import customtkinter as ctk
import numpy as np
from tkinter import messagebox
import threading
import time

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class GaussJordanSolver:

    @staticmethod
    def solve(matrix_a, matrix_b, show_steps=True):
        """
        Solving Matrix Ax = Matrix b (vector) using Gauss-Jordan elimination with step tracking
        Optimized for larger matrices
        """
        try:
            A = np.array(matrix_a, dtype=float).copy()
            b = np.array(matrix_b, dtype=float).copy()
            n = len(b)

            steps = []
            step_num = 0

            # Initial state
            steps.append({
                'step': step_num,
                'desc': 'Initial augmented matrix',
                'A': A.copy(),
                'b': b.copy()
            })
            step_num += 1

            # Track pivot positions for step optimization
            pivot_positions = []

            for k in range(n):
                pivot_row = k

                # Step 1: Partial pivoting with tolerance
                max_val = abs(A[k, k])
                max_row = k
                for i in range(k + 1, n):
                    if abs(A[i, k]) > max_val:
                        max_val = abs(A[i, k])
                        max_row = i

                # Swap if necessary
                if max_row != k and max_val > 1e-12:
                    A[[k, max_row]] = A[[max_row, k]]
                    b[[k, max_row]] = b[[max_row, k]]

                    if show_steps and n <= 15:  # Only record detailed steps for smaller matrices
                        steps.append({
                            'step': step_num,
                            'desc': f'Pivot: Swap row {k + 1} with row {max_row + 1}',
                            'A': A.copy(),
                            'b': b.copy()
                        })
                        step_num += 1

                # Check if pivot is effectively zero
                if abs(A[k, k]) < 1e-12:
                    # Try to find a non-zero pivot in the same row from different column
                    found = False
                    for j in range(k + 1, n):
                        if abs(A[k, j]) > 1e-12:
                            # Swap columns (this changes the variable order)
                            A[:, [k, j]] = A[:, [j, k]]
                            found = True
                            break
                    if not found:
                        continue  # Skip to next row if all zeros

                # Step 2: Normalize pivot row
                pivot = A[k, k]
                if abs(pivot) > 1e-12:
                    A[k] = A[k] / pivot
                    b[k] = b[k] / pivot

                    if show_steps and n <= 15:  # Only record detailed steps for smaller matrices
                        steps.append({
                            'step': step_num,
                            'desc': f'Normalize row {k + 1}: Divide by {pivot:.4f}',
                            'A': A.copy(),
                            'b': b.copy()
                        })
                        step_num += 1

                # Step 3: Eliminate from other rows (vectorized for speed)
                for i in range(n):
                    if i != k and abs(A[i, k]) > 1e-12:
                        factor = A[i, k]
                        A[i] = A[i] - factor * A[k]
                        b[i] = b[i] - factor * b[k]

                        if show_steps and n <= 10:  # Only record elimination steps for small matrices
                            steps.append({
                                'step': step_num,
                                'desc': f'Eliminate: R{i + 1} ← R{i + 1} - ({factor:.4f})×R{k + 1}',
                                'A': A.copy(),
                                'b': b.copy()
                            })
                            step_num += 1

            # Final step (always recorded)
            steps.append({
                'step': step_num,
                'desc': 'Final solution (Reduced Row Echelon Form)',
                'A': A.copy(),
                'b': b.copy()
            })

            return b, A, steps

        except Exception as e:
            raise Exception(f"Solution failed: {str(e)}")

    @staticmethod
    def verify(matrix_a, matrix_b, solution):
        """Verify the solution by checking A*x ≈ b"""
        A = np.array(matrix_a, dtype=float)
        b = np.array(matrix_b, dtype=float)
        x = np.array(solution, dtype=float)

        calculated = A @ x
        error = np.abs(b - calculated)

        return {
            'original': b,
            'calculated': calculated,
            'error': error,
            'max_error': np.max(error),
            'is_correct': np.all(error < 1e-10)
        }


class ScrollableMatrixFrame(ctk.CTkScrollableFrame):
    """Custom scrollable frame for matrix input with optimized performance"""

    def __init__(self, master, rows, cols, entry_width=70, entry_height=35, **kwargs):
        super().__init__(master, **kwargs)
        self.rows = rows
        self.cols = cols
        self.entries = []

        # Create grid of entries
        for i in range(rows):
            row_entries = []
            row_frame = ctk.CTkFrame(self)
            row_frame.grid(row=i, column=0, sticky="w", pady=2)

            for j in range(cols):
                entry = ctk.CTkEntry(
                    row_frame,
                    width=entry_width,
                    height=entry_height,
                    placeholder_text=f"({i},{j})",
                    font=("Consolas", 11)
                )
                entry.grid(row=0, column=j, padx=2)
                row_entries.append(entry)

            self.entries.append(row_entries)


class ScrollableMatrixDisplay(ctk.CTkScrollableFrame):
    """Scrollable display for large matrices in step viewer"""

    def __init__(self, master, matrix, vector, **kwargs):
        super().__init__(master, **kwargs)
        self.matrix = matrix
        self.vector = vector
        self.n = len(vector)
        self.cells = []

        self.create_display()

    def create_display(self):
        # Configure font size based on matrix size
        if self.n <= 8:
            font_size = 12
            cell_width = 90
            cell_height = 40
        elif self.n <= 12:
            font_size = 11
            cell_width = 80
            cell_height = 35
        elif self.n <= 16:
            font_size = 10
            cell_width = 75
            cell_height = 32
        else:
            font_size = 9
            cell_width = 70
            cell_height = 30

        for i in range(self.n):
            row_cells = []
            row_frame = ctk.CTkFrame(self)
            row_frame.grid(row=i, column=0, sticky="w", pady=1)

            # Row label
            label = ctk.CTkLabel(row_frame, text=f"R{i + 1}:", width=40, font=("Consolas", font_size))
            label.grid(row=0, column=0, padx=(0, 5))

            # Matrix values
            for j in range(self.n):
                value = self.matrix[i, j]
                display_value = "0.000" if abs(value) < 1e-10 else f"{value:.4f}"
                text_color = "#888888" if abs(value) < 1e-10 else "#ffffff"

                cell = ctk.CTkLabel(
                    row_frame,
                    text=display_value,
                    width=cell_width,
                    height=cell_height,
                    font=("Consolas", font_size),
                    text_color=text_color,
                    fg_color="#2b2b2b" if abs(value) < 1e-10 else "#3a3a3a",
                    corner_radius=4
                )
                cell.grid(row=0, column=j + 1, padx=1)
                row_cells.append(cell)

            # Separator
            sep = ctk.CTkLabel(row_frame, text="|", font=("Consolas", font_size), width=20)
            sep.grid(row=0, column=self.n + 1, padx=5)

            # Vector value
            b_value = self.vector[i]
            b_display = "0.000" if abs(b_value) < 1e-10 else f"{b_value:.4f}"
            b_cell = ctk.CTkLabel(
                row_frame,
                text=b_display,
                width=cell_width,
                height=cell_height,
                font=("Consolas", font_size),
                text_color="#ffffff",
                fg_color="#1e3a5f",
                corner_radius=4
            )
            b_cell.grid(row=0, column=self.n + 2, padx=1)
            row_cells.append(b_cell)

            self.cells.append(row_cells)


class GaussJordanApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Gauss-Jordan Calculator - Professional Edition")
        self.geometry("1400x900")

        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Variables
        self.matrix_size = ctk.IntVar(value=4)
        self.steps_data = []
        self.current_step = 0
        self.solution = None
        self.solving_thread = None
        self.max_matrix_size = 20  # Maximum allowed matrix size

        # Performance tracking
        self.solve_start_time = 0

        # UI creation
        self.create_widgets()

    def create_widgets(self):
        # --- TOP CONTROL FRAME ---
        control_frame = ctk.CTkFrame(self, height=80)
        control_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        control_frame.grid_propagate(False)

        # Matrix size with range info
        size_frame = ctk.CTkFrame(control_frame)
        size_frame.pack(side="left", padx=(20, 10), pady=20)

        ctk.CTkLabel(size_frame, text="Matrix Size:", font=("Consolas", 14)).pack(side="left", padx=(0, 5))
        self.size_entry = ctk.CTkEntry(size_frame, width=60, textvariable=self.matrix_size,
                                       font=("Consolas", 14), height=35)
        self.size_entry.pack(side="left", padx=5)
        ctk.CTkLabel(size_frame, text=f"(2-{self.max_matrix_size})", font=("Consolas", 12),
                     text_color="gray").pack(side="left", padx=5)

        # Matrix button
        self.create_btn = ctk.CTkButton(
            control_frame,
            text="Create Matrix",
            command=self.create_matrix,
            fg_color="green",
            font=("Consolas", 14),
            height=40,
            width=140
        )
        self.create_btn.pack(side="left", padx=10, pady=20)

        # Solve button
        self.solve_btn = ctk.CTkButton(
            control_frame,
            text="Solve",
            command=self.start_solve_thread,
            fg_color="blue",
            font=("Consolas", 14),
            height=40,
            width=120,
            state="disabled"
        )
        self.solve_btn.pack(side="left", padx=10, pady=20)

        # Step-by-step button
        self.steps_btn = ctk.CTkButton(
            control_frame,
            text="Show Steps",
            command=self.show_steps_viewer,
            fg_color="orange",
            font=("Consolas", 14),
            height=40,
            width=120,
            state="disabled"
        )
        self.steps_btn.pack(side="left", padx=10, pady=20)

        # Clear button
        self.clear_btn = ctk.CTkButton(
            control_frame,
            text="Clear All",
            command=self.clear_all,
            fg_color="red",
            font=("Consolas", 14),
            height=40,
            width=120
        )
        self.clear_btn.pack(side="left", padx=10, pady=20)

        # Status label
        self.status_label = ctk.CTkLabel(
            control_frame,
            text="Ready",
            text_color="gray",
            font=("Consolas", 14)
        )
        self.status_label.pack(side="left", padx=30, pady=20)

        # Performance info label
        self.perf_label = ctk.CTkLabel(
            control_frame,
            text="",
            text_color="blue",
            font=("Consolas", 12)
        )
        self.perf_label.pack(side="right", padx=20, pady=20)

        # --- MAIN CONTENT AREA ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Show welcome screen
        self.show_welcome_screen()

    def show_welcome_screen(self):
        """Display welcome/instructions screen"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        welcome_frame = ctk.CTkFrame(self.main_frame)
        welcome_frame.pack(fill="both", expand=True, padx=60, pady=60)

        # Title
        ctk.CTkLabel(welcome_frame, text="Professional Gauss-Jordan Solver",
                     font=("Consolas", 28, "bold")).pack(pady=40)

        # Matrix size capabilities
        '''
        size_info = ctk.CTkFrame(welcome_frame)
        size_info.pack(pady=20, fill="x", padx=100)

        ctk.CTkLabel(size_info, text="Matrix Size Capabilities:",
                     font=("Consolas", 18, "bold")).pack(pady=10)

        capabilities = [
            f"• Small matrices (2-8): Full step-by-step visualization",
            f"• Medium matrices (9-12): Limited step recording",
            f"• Large matrices (13-{self.max_matrix_size}): Final result only",
            f"• Maximum supported: {self.max_matrix_size}×{self.max_matrix_size} system",
            f"• Performance optimized for up to {self.max_matrix_size}×{self.max_matrix_size}"
        ]

        for capability in capabilities:
            ctk.CTkLabel(size_info, text=capability,
                         font=("Consolas", 14), justify="left").pack(pady=2, anchor="w")
        '''

        # Instructions
        instructions_frame = ctk.CTkFrame(welcome_frame)
        instructions_frame.pack(pady=30, fill="x", padx=100)

        ctk.CTkLabel(instructions_frame, text="Instructions:",
                     font=("Consolas", 18, "bold")).pack(pady=10)

        instructions = [
            "1. Enter matrix size between 2 and 20",
            "2. Click 'Create Matrix' to generate input fields",
            "3. Fill in coefficient matrix A and constant vector b",
            "4. Click 'Solve' to compute solution",
            "5. View step-by-step for small/medium matrices"
        ]

        for instruction in instructions:
            ctk.CTkLabel(instructions_frame, text=instruction,
                         font=("Consolas", 14)).pack(pady=5, anchor="w")

        '''
        # Performance tips
        tips_frame = ctk.CTkFrame(welcome_frame)
        tips_frame.pack(pady=30, fill="x", padx=100)

        ctk.CTkLabel(tips_frame, text="Performance Tips:",
                     font=("Consolas", 18, "bold")).pack(pady=10)

        tips = [
            "• For matrices > 12×12, step-by-step is disabled (too many steps)",
            "• Solutions are computed efficiently using optimized numpy operations",
            "• Verification checks solution accuracy automatically",
            "• Use 0 for empty cells (they default to 0)"
        ]

        for tip in tips:
            ctk.CTkLabel(tips_frame, text=tip,
                         font=("Consolas", 14), justify="left").pack(pady=2, anchor="w")

        # Quick start buttons
        btn_frame = ctk.CTkFrame(welcome_frame)
        btn_frame.pack(pady=40)

        # Example buttons for different sizes
        sizes = [("3×3 Example", 3), ("5×5 Example", 5), ("8×8 Example", 8), ("12×12 Example", 12)]

        for text, size in sizes:
            btn = ctk.CTkButton(
                btn_frame,
                text=text,
                command=lambda s=size: self.load_example_size(s),
                fg_color="orange",
                font=("Consolas", 14),
                height=40,
                width=150
            )
            btn.pack(side="left", padx=10)

    def load_example_size(self, size):
        """Load example of specific size"""
        self.matrix_size.set(size)
        self.create_matrix()

        # Fill with example values for common sizes
        if size == 3:
            # Classic 3x3 example
            example_A = [[2, 1, -1],
                         [-3, -1, 2],
                         [-2, 1, 2]]
            example_b = [8, -11, -3]
            self.fill_example_values(example_A, example_b)

        elif size == 4:
            # 4x4 example
            example_A = [[4, -1, 0, 3],
                         [2, 5, -2, 1],
                         [1, -1, 3, -2],
                         [3, 2, -1, 4]]
            example_b = [10, 8, 5, 12]
            self.fill_example_values(example_A, example_b)

        elif size == 5:
            # 5x5 identity-like with small perturbations
            example_A = [[1, 0.1, 0.2, 0.1, 0.3],
                         [0.2, 1, 0.1, 0.2, 0.1],
                         [0.1, 0.3, 1, 0.1, 0.2],
                         [0.3, 0.1, 0.2, 1, 0.1],
                         [0.2, 0.2, 0.1, 0.3, 1]]
            example_b = [1.7, 1.6, 1.7, 1.7, 1.8]
            self.fill_example_values(example_A, example_b)

    def fill_example_values(self, matrix_A, vector_b):
        """Fill the input fields with example values"""
        size = len(vector_b)
        for i in range(size):
            for j in range(size):
                if i < len(matrix_A) and j < len(matrix_A[i]):
                    self.a_entries[i][j].delete(0, "end")
                    self.a_entries[i][j].insert(0, str(matrix_A[i][j]))

        for i in range(size):
            if i < len(vector_b):
                self.b_entries[i].delete(0, "end")
                self.b_entries[i].insert(0, str(vector_b[i]))
        '''

    def create_matrix(self):
        try:
            size = self.matrix_size.get()
            if size < 2 or size > self.max_matrix_size:
                messagebox.showerror("Error", f"Matrix size must be between 2 and {self.max_matrix_size}")
                return

            # Clear main frame
            for widget in self.main_frame.winfo_children():
                widget.destroy()

            # Create main container with scroll if needed
            if size > 10:
                # Use scrollable frames for large matrices
                main_container = ctk.CTkScrollableFrame(self.main_frame, height=600)
                main_container.pack(fill="both", expand=True, padx=10, pady=10)
                content_frame = main_container
            else:
                content_frame = ctk.CTkFrame(self.main_frame)
                content_frame.pack(fill="both", expand=True, padx=30, pady=30)

            # Title with size warning
            title_text = f"Enter {size}×{size} System"
            if size > 12:
                title_text += " (Large System - Step-by-Step Disabled)"

            title_label = ctk.CTkLabel(content_frame, text=title_text,
                                       font=("Consolas", 18, "bold"))
            title_label.pack(pady=15)

            # Matrix input area
            matrix_area = ctk.CTkFrame(content_frame)
            matrix_area.pack(pady=20, fill="x")

            # Left: Matrix A with scroll if large
            if size > 8:
                a_container = ctk.CTkScrollableFrame(matrix_area, width=400, height=300)
                a_container.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
                a_frame = a_container
            else:
                a_frame = ctk.CTkFrame(matrix_area)
                a_frame.grid(row=0, column=0, padx=20, pady=10)

            ctk.CTkLabel(a_frame, text="Matrix A",
                         font=("Consolas", 14, "bold")).pack(pady=5)

            # Create A entries grid
            self.a_entries = []
            entry_width = 70 if size <= 10 else 60
            entry_height = 35 if size <= 10 else 30
            font_size = 12 if size <= 10 else 11

            for i in range(size):
                row_frame = ctk.CTkFrame(a_frame)
                row_frame.pack(pady=2)
                row_entries = []

                for j in range(size):
                    entry = ctk.CTkEntry(
                        row_frame,
                        width=entry_width,
                        height=entry_height,
                        placeholder_text="0",
                        font=("Consolas", font_size)
                    )
                    entry.pack(side="left", padx=2)
                    row_entries.append(entry)
                self.a_entries.append(row_entries)

            # Center: Symbols
            sym_frame = ctk.CTkFrame(matrix_area)
            sym_frame.grid(row=0, column=1, padx=10, pady=10)

            ctk.CTkLabel(sym_frame, text="×", font=("Consolas", 24)).pack(pady=15)
            ctk.CTkLabel(sym_frame, text="X", font=("Consolas", 24)).pack(pady=15)
            ctk.CTkLabel(sym_frame, text="=", font=("Consolas", 24)).pack(pady=15)

            # Right: Vector b with scroll if large
            if size > 12:
                b_container = ctk.CTkScrollableFrame(matrix_area, width=200, height=300)
                b_container.grid(row=0, column=2, padx=20, pady=10, sticky="nsew")
                b_frame = b_container
            else:
                b_frame = ctk.CTkFrame(matrix_area)
                b_frame.grid(row=0, column=2, padx=20, pady=10)

            ctk.CTkLabel(b_frame, text="Vector b",
                         font=("Consolas", 14, "bold")).pack(pady=5)

            # Create b entries
            self.b_entries = []
            for i in range(size):
                entry = ctk.CTkEntry(
                    b_frame,
                    width=entry_width,
                    height=entry_height,
                    placeholder_text=f"b{i + 1}",
                    font=("Consolas", font_size)
                )
                entry.pack(pady=2)
                self.b_entries.append(entry)

            # Button frame
            btn_frame = ctk.CTkFrame(content_frame)
            btn_frame.pack(pady=30)

            # Solve button
            self.solve_btn.configure(state="normal")
            solve_btn = ctk.CTkButton(
                btn_frame,
                text="Solve System",
                command=self.start_solve_thread,
                fg_color="green",
                width=150,
                height=40,
                font=("Consolas", 14)
            )
            solve_btn.pack(side="left", padx=15)

            # Random fill button (useful for testing large matrices)
            random_btn = ctk.CTkButton(
                btn_frame,
                text="Fill Random",
                command=self.fill_random_values,
                fg_color="purple",
                width=150,
                height=40,
                font=("Consolas", 14)
            )
            random_btn.pack(side="left", padx=15)

            # Clear inputs button
            clear_inputs_btn = ctk.CTkButton(
                btn_frame,
                text="Clear Inputs",
                command=self.clear_inputs,
                fg_color="gray",
                width=150,
                height=40,
                font=("Consolas", 14)
            )
            clear_inputs_btn.pack(side="left", padx=15)

            self.status_label.configure(text=f"Created {size}×{size} system", text_color="green")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create matrix: {str(e)}")

    def fill_random_values(self):
        """Fill matrix with random values for testing"""
        try:
            size = self.matrix_size.get()

            # Generate random matrix A (mostly diagonally dominant for solvability)
            for i in range(size):
                for j in range(size):
                    if i == j:
                        # Diagonal elements: larger for stability
                        value = np.random.uniform(1, 5)
                    else:
                        # Off-diagonal: smaller random values
                        value = np.random.uniform(-1, 1)

                    self.a_entries[i][j].delete(0, "end")
                    self.a_entries[i][j].insert(0, f"{value:.2f}")

            # Generate random vector b
            for i in range(size):
                value = np.random.uniform(-10, 10)
                self.b_entries[i].delete(0, "end")
                self.b_entries[i].insert(0, f"{value:.2f}")

            self.status_label.configure(text=f"Filled with random values", text_color="blue")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fill random values: {str(e)}")

    def clear_inputs(self):
        """Clear all input fields"""
        try:
            size = self.matrix_size.get()
            for i in range(size):
                for j in range(size):
                    self.a_entries[i][j].delete(0, "end")
                self.b_entries[i].delete(0, "end")

            self.status_label.configure(text="Inputs cleared", text_color="gray")

        except:
            pass

    def get_matrix_values(self):
        try:
            size = self.matrix_size.get()
            matrix_a = []
            matrix_b = []

            # Get matrix A
            for i in range(size):
                row = []
                for j in range(size):
                    val = self.a_entries[i][j].get()
                    if val == "":
                        val = "0"
                    try:
                        row.append(float(val))
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid value at A[{i + 1},{j + 1}]: '{val}'")
                        return None, None
                matrix_a.append(row)

            # Get vector b
            for i in range(size):
                val = self.b_entries[i].get()
                if val == "":
                    val = "0"
                try:
                    matrix_b.append(float(val))
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value at b[{i + 1}]: '{val}'")
                    return None, None

            return matrix_a, matrix_b

        except Exception as e:
            messagebox.showerror("Error", f"Error reading values: {str(e)}")
            return None, None

    def start_solve_thread(self):
        """Start solving in a separate thread to keep GUI responsive"""
        if self.solving_thread and self.solving_thread.is_alive():
            return

        self.solving_thread = threading.Thread(target=self.solve_system_thread, daemon=True)
        self.solving_thread.start()

    def solve_system_thread(self):
        """Solve system in background thread"""
        matrix_a, matrix_b = self.get_matrix_values()
        if matrix_a is None:
            return

        try:
            # Update UI in main thread
            self.after(0, lambda: self.status_label.configure(text="Solving...", text_color="orange"))
            self.after(0, self.update)

            # Record start time
            self.solve_start_time = time.time()

            # Create solver instance
            solver = GaussJordanSolver()

            # Solve the system with appropriate step detail
            size = len(matrix_b)
            show_detailed_steps = size <= 12  # Only show detailed steps for smaller matrices

            solution, rref, steps = solver.solve(matrix_a, matrix_b, show_detailed_steps)

            # Calculate solve time
            solve_time = time.time() - self.solve_start_time

            # Store for later use
            self.solution = solution
            self.steps_data = steps
            self.matrix_a = matrix_a
            self.matrix_b = matrix_b
            self.current_step = 0

            # Update UI in main thread
            self.after(0, lambda: self.show_solution(solution, matrix_a, matrix_b, solve_time))
            self.after(0, lambda: self.status_label.configure(
                text=f"Solved in {solve_time:.3f}s",
                text_color="green"
            ))

            # Enable steps button only for smaller matrices
            if size <= 12:
                self.after(0, lambda: self.steps_btn.configure(state="normal"))
            else:
                self.after(0, lambda: self.steps_btn.configure(state="disabled"))

        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text="Error", text_color="red"))
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to solve system: {str(e)}"))

    def show_solution(self, solution, matrix_a, matrix_b, solve_time=0):
        """Display the solution"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        size = len(solution)

        # Use scrollable frame for large solutions
        if size > 10:
            solution_frame = ctk.CTkScrollableFrame(self.main_frame, height=600)
        else:
            solution_frame = ctk.CTkFrame(self.main_frame)

        solution_frame.pack(fill="both", expand=True, padx=30, pady=30)

        # Header
        header_frame = ctk.CTkFrame(solution_frame)
        header_frame.pack(fill="x", pady=15)

        ctk.CTkLabel(header_frame, text="Solution",
                     font=("Consolas", 24, "bold")).pack(side="left", padx=15)

        # Performance info
        if solve_time > 0:
            perf_text = f"Solve time: {solve_time:.3f}s | Size: {size}×{size}"
            ctk.CTkLabel(header_frame, text=perf_text,
                         font=("Consolas", 12), text_color="blue").pack(side="left", padx=20)

        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back",
            command=self.create_matrix,
            fg_color="gray",
            width=100,
            height=40,
            font=("Consolas", 14)
        )
        back_btn.pack(side="right", padx=15)

        # Solution values
        ctk.CTkLabel(solution_frame, text="Solution Vector X:",
                     font=("Consolas", 18)).pack(pady=20)

        # Display solutions in a grid for large matrices
        if size > 8:
            # Use grid layout for large solutions
            solutions_frame = ctk.CTkFrame(solution_frame)
            solutions_frame.pack(pady=10, fill="x")

            cols = 4  # Number of columns in grid
            rows = (size + cols - 1) // cols

            for i in range(size):
                row = i // cols
                col = i % cols

                sol_frame = ctk.CTkFrame(solutions_frame)
                sol_frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")

                ctk.CTkLabel(sol_frame, text=f"x{i + 1} = ",
                             font=("Consolas", 14)).pack(side="left", padx=5)

                value_label = ctk.CTkLabel(
                    sol_frame,
                    text=f"{solution[i]:.6f}",
                    font=("Courier", 14, "bold"),
                    text_color="green"
                )
                value_label.pack(side="left", padx=5)
        else:
            # Vertical layout for smaller solutions
            for i in range(size):
                row_frame = ctk.CTkFrame(solution_frame)
                row_frame.pack(pady=5)

                ctk.CTkLabel(row_frame, text=f"x{i + 1} = ",
                             font=("Consolas", 16)).pack(side="left", padx=5)

                value_label = ctk.CTkLabel(
                    row_frame,
                    text=f"{solution[i]:.8f}",
                    font=("Courier", 16, "bold"),
                    text_color="green"
                )
                value_label.pack(side="left", padx=5)

        # Button frame
        btn_frame = ctk.CTkFrame(solution_frame)
        btn_frame.pack(pady=30)

        # Verify button
        verify_btn = ctk.CTkButton(
            btn_frame,
            text="Verify Solution",
            command=self.verify_solution,
            fg_color="blue",
            width=150,
            height=40,
            font=("Consolas", 14)
        )
        verify_btn.pack(side="left", padx=10)

        # Steps button (only for smaller matrices)
        if size <= 12:
            self.steps_btn = ctk.CTkButton(
                btn_frame,
                text="View Step-by-Step",
                command=self.show_steps_viewer,
                fg_color="orange",
                width=150,
                height=40,
                font=("Consolas", 14)
            )
            self.steps_btn.pack(side="left", padx=10)
        else:
            # Info label for large matrices
            info_label = ctk.CTkLabel(
                btn_frame,
                text="Step-by-Step disabled for large matrices",
                text_color="gray",
                font=("Consolas", 12)
            )
            info_label.pack(side="left", padx=10)

        # New problem button
        new_btn = ctk.CTkButton(
            btn_frame,
            text="New Problem",
            command=self.clear_all,
            fg_color="green",
            width=150,
            height=40,
            font=("Consolas", 14)
        )
        new_btn.pack(side="left", padx=10)

    def verify_solution(self):
        if self.solution is None:
            return

        try:
            solver = GaussJordanSolver()
            result = solver.verify(self.matrix_a, self.matrix_b, self.solution)

            # Create verification window
            verify_window = ctk.CTkToplevel(self)
            verify_window.title("Solution Verification")

            # Adjust window size based on matrix size
            size = len(self.solution)
            window_height = min(600, 200 + size * 30)
            verify_window.geometry(f"700x{window_height}")

            # Title
            ctk.CTkLabel(verify_window, text="Verification Results",
                         font=("Consolas", 20, "bold")).pack(pady=20)

            # Create scrollable content
            scroll_frame = ctk.CTkScrollableFrame(verify_window)
            scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

            # Original b
            ctk.CTkLabel(scroll_frame, text="Original b:",
                         font=("Consolas", 14, "bold")).pack(anchor="w", pady=5)

            # Display b values in compact format for large matrices
            if size > 10:
                # Display as array
                orig_array = np.array2string(result['original'], precision=6, suppress_small=True)
                ctk.CTkLabel(scroll_frame, text=orig_array,
                             font=("Courier", 11), justify="left").pack(anchor="w", pady=2)
            else:
                orig_text = ""
                for i, val in enumerate(result['original']):
                    orig_text += f"b{i + 1} = {val:12.6f}\n"
                ctk.CTkLabel(scroll_frame, text=orig_text,
                             font=("Courier", 12), justify="left").pack(anchor="w", pady=2)

            # Calculated A*x
            ctk.CTkLabel(scroll_frame, text="Calculated A*x:",
                         font=("Consolas", 14, "bold")).pack(anchor="w", pady=(15, 5))

            if size > 10:
                calc_array = np.array2string(result['calculated'], precision=6, suppress_small=True)
                ctk.CTkLabel(scroll_frame, text=calc_array,
                             font=("Courier", 11), justify="left").pack(anchor="w", pady=2)
            else:
                calc_text = ""
                for i, val in enumerate(result['calculated']):
                    calc_text += f"b{i + 1} = {val:12.6f}\n"
                ctk.CTkLabel(scroll_frame, text=calc_text,
                             font=("Courier", 12), justify="left").pack(anchor="w", pady=2)

            # Max error
            ctk.CTkLabel(scroll_frame,
                         text=f"Maximum Error: {result['max_error']:.2e}",
                         font=("Consolas", 14, "bold")).pack(anchor="w", pady=20)

            # Conclusion
            if result['is_correct']:
                ctk.CTkLabel(scroll_frame, text="✓ Solution is CORRECT",
                             text_color="green", font=("Consolas", 16, "bold")).pack(pady=10)
            else:
                ctk.CTkLabel(scroll_frame, text="✗ Solution has ERRORS",
                             text_color="red", font=("Consolas", 16, "bold")).pack(pady=10)

            # Close button
            ctk.CTkButton(verify_window, text="Close",
                          command=verify_window.destroy,
                          fg_color="gray",
                          width=120,
                          height=40,
                          font=("Consolas", 14)).pack(pady=20)

        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

    def show_steps_viewer(self):
        """Show step-by-step solution viewer"""
        if not self.steps_data:
            messagebox.showinfo("No Steps", "No step-by-step data available")
            return

        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create steps viewer
        steps_frame = ctk.CTkFrame(self.main_frame)
        steps_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        header_frame = ctk.CTkFrame(steps_frame)
        header_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(header_frame, text="Step-by-Step Solution",
                     font=("Consolas", 20, "bold")).pack(side="left", padx=15)

        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back to Solution",
            command=lambda: self.show_solution(self.solution, self.matrix_a, self.matrix_b, 0),
            fg_color="gray",
            width=160,
            height=40,
            font=("Consolas", 14)
        )
        back_btn.pack(side="right", padx=15)

        # Navigation controls
        nav_frame = ctk.CTkFrame(steps_frame)
        nav_frame.pack(fill="x", pady=10)

        self.prev_btn = ctk.CTkButton(
            nav_frame,
            text="◀ Previous",
            command=self.prev_step,
            state="disabled",
            width=140,
            height=40,
            font=("Consolas", 14)
        )
        self.prev_btn.pack(side="left", padx=15)

        self.step_label = ctk.CTkLabel(
            nav_frame,
            text="Step 1/1",
            font=("Consolas", 16)
        )
        self.step_label.pack(side="left", padx=25)

        self.next_btn = ctk.CTkButton(
            nav_frame,
            text="Next ▶",
            command=self.next_step,
            state="normal",
            width=140,
            height=40,
            font=("Consolas", 14)
        )
        self.next_btn.pack(side="left", padx=15)

        # Jump to specific step for large step counts
        if len(self.steps_data) > 20:
            jump_frame = ctk.CTkFrame(nav_frame)
            jump_frame.pack(side="right", padx=15)

            ctk.CTkLabel(jump_frame, text="Go to:", font=("Consolas", 12)).pack(side="left", padx=5)
            self.step_entry = ctk.CTkEntry(jump_frame, width=50, font=("Consolas", 12))
            self.step_entry.pack(side="left", padx=5)

            jump_btn = ctk.CTkButton(
                jump_frame,
                text="Go",
                command=self.jump_to_step,
                width=50,
                height=30,
                font=("Consolas", 12)
            )
            jump_btn.pack(side="left", padx=5)

        # Main content area
        content_frame = ctk.CTkFrame(steps_frame)
        content_frame.pack(fill="both", expand=True, pady=10)

        # Left side: Step description
        desc_frame = ctk.CTkFrame(content_frame, width=350)
        desc_frame.pack(side="left", fill="y", padx=(0, 20))

        self.desc_label = ctk.CTkLabel(
            desc_frame,
            text="",
            font=("Consolas", 16, "bold"),
            wraplength=330,
            justify="left"
        )
        self.desc_label.pack(pady=15, padx=15)

        # Operation details
        self.op_details = ctk.CTkTextbox(
            desc_frame,
            height=180,
            font=("Consolas", 13),
            wrap="word"
        )
        self.op_details.pack(fill="both", expand=True, padx=15, pady=15)
        self.op_details.configure(state="disabled")

        # Right side: Matrix display
        matrix_display_frame = ctk.CTkFrame(content_frame)
        matrix_display_frame.pack(side="right", fill="both", expand=True)

        # Matrix display area
        self.matrix_display_area = ctk.CTkScrollableFrame(matrix_display_frame)
        self.matrix_display_area.pack(fill="both", expand=True, padx=15, pady=15)

        # Show first step
        self.current_step = 0
        self.show_current_step()

    def show_current_step(self):
        """Display the current step"""
        if not self.steps_data or self.current_step >= len(self.steps_data):
            return

        step = self.steps_data[self.current_step]

        # Clear matrix display area
        for widget in self.matrix_display_area.winfo_children():
            widget.destroy()

        # Update description
        self.desc_label.configure(text=f"Step {step['step']}: {step['desc']}")

        # Update operation details
        self.op_details.configure(state="normal")
        self.op_details.delete("1.0", "end")

        # Add operation explanation
        desc = step['desc']
        operation_details = {
            'Swap': "Row swapping improves numerical stability by ensuring the pivot element is the largest in its column.",
            'Normalize': "Row normalization sets the pivot element to 1, simplifying subsequent elimination steps.",
            'Eliminate': "Row elimination creates zeros in the pivot column, progressively forming an identity matrix.",
            'Initial': "Starting augmented matrix [A|b] before any operations.",
            'Final': "Reduced Row Echelon Form (RREF) where A is identity matrix and b contains the solution."
        }

        for key, explanation in operation_details.items():
            if key in desc:
                self.op_details.insert("1.0", explanation)
                break
        else:
            self.op_details.insert("1.0", "Matrix transformation step.")

        self.op_details.configure(state="disabled")

        # Display matrix
        try:
            A = step['A']
            b = step['b']

            # Create scrollable matrix display
            matrix_display = ScrollableMatrixDisplay(
                self.matrix_display_area,
                A,
                b
            )
            matrix_display.pack(fill="both", expand=True)

        except Exception as e:
            error_label = ctk.CTkLabel(
                self.matrix_display_area,
                text=f"Error displaying matrix: {str(e)}",
                font=("Consolas", 14),
                text_color="red"
            )
            error_label.pack(expand=True)

        # Update navigation
        self.step_label.configure(text=f"Step {self.current_step + 1}/{len(self.steps_data)}")
        self.prev_btn.configure(state="normal" if self.current_step > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_step < len(self.steps_data) - 1 else "disabled")

    def jump_to_step(self):
        """Jump to a specific step number"""
        try:
            step_num = int(self.step_entry.get()) - 1
            if 0 <= step_num < len(self.steps_data):
                self.current_step = step_num
                self.show_current_step()
        except:
            pass

    def next_step(self):
        """Go to next step"""
        if self.current_step < len(self.steps_data) - 1:
            self.current_step += 1
            self.show_current_step()

    def prev_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.show_current_step()

    def clear_all(self):
        """Clear everything and show welcome screen"""
        self.matrix_size.set(4)
        self.steps_data = []
        self.current_step = 0
        self.solution = None
        self.solve_btn.configure(state="disabled")
        self.steps_btn.configure(state="disabled")
        self.status_label.configure(text="Ready", text_color="gray")
        self.perf_label.configure(text="")
        self.show_welcome_screen()


if __name__ == "__main__":
    app = GaussJordanApp()
    app.mainloop()
