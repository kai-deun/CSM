import customtkinter as ctk
import numpy as np
from tkinter import messagebox

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

'''
One application based on the console-based program of Iris
and Initial design of Jovany for the GUI
'''


class GaussJordanSolver:

    @staticmethod
    def solve(matrix_a, matrix_b, show_steps=True):
        """
        Solving Matrix Ax = Matrix b (vector) using Gauss-Jordan elimination with step tracking
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
                'b': b.copy(),
                'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
            })
            step_num += 1

            for k in range(n):
                # Step 1: Partial pivoting if needed
                if abs(A[k, k]) < 1e-10:
                    for i in range(k + 1, n):
                        if abs(A[i, k]) > abs(A[k, k]):
                            # Swap rows
                            A[[k, i]] = A[[i, k]]
                            b[[k, i]] = b[[i, k]]

                            steps.append({
                                'step': step_num,
                                'desc': f'Pivot: Swap row {k + 1} with row {i + 1}',
                                'A': A.copy(),
                                'b': b.copy(),
                                'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
                            })
                            step_num += 1
                            break

                # Step 2: Normalize pivot row
                pivot = A[k, k]
                if abs(pivot) > 1e-12:
                    A[k] = A[k] / pivot
                    b[k] = b[k] / pivot

                    steps.append({
                        'step': step_num,
                        'desc': f'Normalize row {k + 1}: Divide by {pivot:.4f}',
                        'A': A.copy(),
                        'b': b.copy(),
                        'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
                    })
                    step_num += 1

                # Step 3: Eliminate from other rows
                for i in range(n):
                    if i != k and abs(A[i, k]) > 1e-12:
                        factor = A[i, k]
                        A[i] = A[i] - factor * A[k]
                        b[i] = b[i] - factor * b[k]

                        steps.append({
                            'step': step_num,
                            'desc': f'Eliminate: R{i + 1} ← R{i + 1} - ({factor:.4f})×R{k + 1}',
                            'A': A.copy(),
                            'b': b.copy(),
                            'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
                        })
                        step_num += 1

            # Final step
            steps.append({
                'step': step_num,
                'desc': 'Final solution (Reduced Row Echelon Form)',
                'A': A.copy(),
                'b': b.copy(),
                'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
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


class MatrixCell(ctk.CTkFrame):
    """Custom widget for displaying a matrix cell - ENLARGED VERSION"""

    def __init__(self, master, value=0, width=120, height=60, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.configure(corner_radius=8, border_width=1, border_color="#555555")

        # Configure padding
        cell_padding = 5

        # Format value based on magnitude
        if abs(value) < 1e-10:
            display_value = "0.000000"
            text_color = "#888888"
        else:
            display_value = f"{value:.6f}"
            text_color = "#ffffff"

        self.value_label = ctk.CTkLabel(
            self,
            text=display_value,
            font=("Consolas", 16, "bold"),
            text_color=text_color
        )
        self.value_label.pack(expand=True, fill="both", padx=cell_padding, pady=cell_padding)

    def update_value(self, value):
        # Format value based on magnitude
        if abs(value) < 1e-10:
            display_value = "0.000000"
            text_color = "#888888"
            bg_color = "#2b2b2b"
        else:
            display_value = f"{value:.6f}"
            text_color = "#ffffff"
            bg_color = "#3a3a3a"

        self.value_label.configure(text=display_value, text_color=text_color)
        self.configure(fg_color=bg_color)


class MatrixDisplay(ctk.CTkFrame):
    """Visual matrix display with cell-like layout - ENLARGED VERSION"""

    def __init__(self, master, matrix, vector, title="Augmented Matrix", **kwargs):
        super().__init__(master, **kwargs)

        # Title with larger font
        title_label = ctk.CTkLabel(self, text=title, font=("Consolas", 16, "bold"))
        title_label.pack(pady=10)

        # Create matrix frame with more padding
        matrix_frame = ctk.CTkFrame(self)
        matrix_frame.pack(pady=15, padx=10)

        n = len(vector)
        self.cells = []

        # Create matrix cells with larger dimensions
        for i in range(n):
            row_cells = []
            row_frame = ctk.CTkFrame(matrix_frame)
            row_frame.pack(pady=5)

            # Matrix A cells - ENLARGED
            for j in range(n):
                cell = MatrixCell(row_frame, matrix[i, j], width=100, height=50)
                cell.pack(side="left", padx=3)
                row_cells.append(cell)

            # Separator - larger and more prominent
            sep = ctk.CTkLabel(
                row_frame,
                text="│",
                font=("Consolas", 20, "bold"),
                width=30,
                text_color="#cccccc"
            )
            sep.pack(side="left", padx=8)

            # Vector b cell - ENLARGED
            b_cell = MatrixCell(row_frame, vector[i], width=100, height=50)
            b_cell.pack(side="left", padx=3)
            row_cells.append(b_cell)

            self.cells.append(row_cells)

    def update_matrix(self, matrix, vector):
        """Update the displayed matrix values"""
        n = len(vector)
        for i in range(n):
            for j in range(n):
                self.cells[i][j].update_value(matrix[i, j])
            # Update b value
            self.cells[i][-1].update_value(vector[i])


class GaussJordanApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Gauss-Jordan Calculator")
        self.geometry("1400x900")  # Increased window size

        # grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # var
        self.matrix_size = ctk.IntVar(value=3)
        self.steps_data = []
        self.current_step = 0
        self.solution = None

        # ui creation
        self.create_widgets()

        # example data
        self.after(100, self.set_default_example)

    def create_widgets(self):
        # --- TOP CONTROL FRAME ---
        control_frame = ctk.CTkFrame(self, height=70)
        control_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        control_frame.grid_propagate(False)

        # matrix size - larger font
        ctk.CTkLabel(control_frame, text="Matrix Size (n×n):", font=("Consolas", 14)).pack(side="left", padx=(20, 10),
                                                                                           pady=20)
        self.size_entry = ctk.CTkEntry(control_frame, width=80, textvariable=self.matrix_size, font=("Consolas", 14),
                                       height=35)
        self.size_entry.pack(side="left", padx=10, pady=20)

        # matrix button - larger
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

        # solve button - larger
        self.solve_btn = ctk.CTkButton(
            control_frame,
            text="Solve",
            command=self.solve_system,
            fg_color="blue",
            font=("Consolas", 14),
            height=40,
            width=120,
            state="disabled"
        )
        self.solve_btn.pack(side="left", padx=10, pady=20)

        # clear button - larger
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

        # status label - larger font
        self.status_label = ctk.CTkLabel(
            control_frame,
            text="Ready",
            text_color="gray",
            font=("Consolas", 14)
        )
        self.status_label.pack(side="left", padx=30, pady=20)

        # --- MAIN CONTENT AREA ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # welcome screen
        self.show_welcome_screen()

    def show_welcome_screen(self):
        """Display welcome/instructions screen"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        welcome_frame = ctk.CTkFrame(self.main_frame)
        welcome_frame.pack(fill="both", expand=True, padx=60, pady=60)

        # Title - larger
        ctk.CTkLabel(welcome_frame, text="Gauss-Jordan Elimination Solver",
                     font=("Consolas", 28, "bold")).pack(pady=40)

        # Instructions - larger font
        instructions = [
            "1. Enter the square matrix size (example.g., 3 for 3×3)",
            "2. Click 'Create Matrix' to generate input fields",
            "3. Enter values for the coefficient matrix and constant matrix",
            "4. Click 'Solve' to compute solution",
            "5. View step-by-step solution with visual matrix display"
        ]

        for instruction in instructions:
            ctk.CTkLabel(welcome_frame, text=instruction,
                         font=("Consolas", 16)).pack(pady=8)

        # Example frame
        example_frame = ctk.CTkFrame(welcome_frame)
        example_frame.pack(pady=40, fill="x", padx=60)

        ctk.CTkLabel(example_frame, text="Example 3×3 System:",
                     font=("Consolas", 18, "bold")).pack(pady=15)

        example = """2x + y - z = 8
-3x - y + 2z = -11
-2x + y + 2z = -3

Solution: x = 2, y = 3, z = -1"""

        ctk.CTkLabel(example_frame, text=example,
                     font=("Courier", 14), justify="left").pack(pady=15)

        # Quick start button - larger
        ctk.CTkButton(
            welcome_frame,
            text="Load Example",
            command=self.load_example,
            fg_color="orange",
            font=("Consolas", 14),
            height=45,
            width=180
        ).pack(pady=30)

    def set_default_example(self):
        self.matrix_size.set(3)
        self.create_matrix()

    def load_example(self):
        self.matrix_size.set(3)
        self.create_matrix()

    def create_matrix(self):
        try:
            size = self.matrix_size.get()
            if size < 2 or size > 8:
                messagebox.showerror("Error", "Matrix size must be between 2 and 8")
                return

            # Clear main frame
            for widget in self.main_frame.winfo_children():
                widget.destroy()

            # Create input area
            input_frame = ctk.CTkFrame(self.main_frame)
            input_frame.pack(fill="both", expand=True, padx=30, pady=30)

            # Title - larger
            ctk.CTkLabel(input_frame, text=f"Enter {size}×{size} System",
                         font=("Consolas", 20, "bold")).pack(pady=15)

            # Matrix input area
            matrix_area = ctk.CTkFrame(input_frame)
            matrix_area.pack(pady=30)

            # Matrix A frame
            a_frame = ctk.CTkFrame(matrix_area)
            a_frame.grid(row=0, column=0, padx=30, pady=20)

            ctk.CTkLabel(a_frame, text="Matrix A",
                         font=("Consolas", 16, "bold")).pack(pady=10)

            # Create A entries grid - ENLARGED
            self.a_entries = []
            for i in range(size):
                row_frame = ctk.CTkFrame(a_frame)
                row_frame.pack(pady=5)
                row_entries = []

                for j in range(size):
                    entry = ctk.CTkEntry(
                        row_frame,
                        width=100,  # Wider
                        height=40,  # Taller
                        placeholder_text="0",
                        font=("Consolas", 14)
                    )
                    # Set default values for example
                    if size == 3:
                        if i == 0 and j == 0:
                            entry.insert(0, "2")
                        elif i == 0 and j == 1:
                            entry.insert(0, "1")
                        elif i == 0 and j == 2:
                            entry.insert(0, "-1")
                        elif i == 1 and j == 0:
                            entry.insert(0, "-3")
                        elif i == 1 and j == 1:
                            entry.insert(0, "-1")
                        elif i == 1 and j == 2:
                            entry.insert(0, "2")
                        elif i == 2 and j == 0:
                            entry.insert(0, "-2")
                        elif i == 2 and j == 1:
                            entry.insert(0, "1")
                        elif i == 2 and j == 2:
                            entry.insert(0, "2")

                    entry.pack(side="left", padx=5)
                    row_entries.append(entry)
                self.a_entries.append(row_entries)

            # Multiplication and equals symbols - larger
            sym_frame = ctk.CTkFrame(matrix_area)
            sym_frame.grid(row=0, column=1, padx=20, pady=20)

            ctk.CTkLabel(sym_frame, text="×", font=("Consolas", 32)).pack(pady=25)
            ctk.CTkLabel(sym_frame, text="X", font=("Consolas", 32)).pack(pady=25)
            ctk.CTkLabel(sym_frame, text="=", font=("Consolas", 32)).pack(pady=25)

            # Vector X labels - larger
            x_frame = ctk.CTkFrame(matrix_area)
            x_frame.grid(row=0, column=2, padx=20, pady=20)

            ctk.CTkLabel(x_frame, text="Vector X",
                         font=("Consolas", 16, "bold")).pack(pady=10)

            for i in range(size):
                ctk.CTkLabel(x_frame, text=f"x{i + 1}",
                             font=("Consolas", 16), width=100, height=45).pack(pady=5)

            # Vector b frame
            b_frame = ctk.CTkFrame(matrix_area)
            b_frame.grid(row=0, column=3, padx=30, pady=20)

            ctk.CTkLabel(b_frame, text="Vector b",
                         font=("Consolas", 16, "bold")).pack(pady=10)

            # Create b entries - ENLARGED
            self.b_entries = []
            for i in range(size):
                entry = ctk.CTkEntry(
                    b_frame,
                    width=100,  # Wider
                    height=40,  # Taller
                    placeholder_text="0",
                    font=("Consolas", 14)
                )
                # Set default values for example
                if size == 3:
                    if i == 0:
                        entry.insert(0, "8")
                    elif i == 1:
                        entry.insert(0, "-11")
                    else:
                        entry.insert(0, "-3")

                entry.pack(pady=5)
                self.b_entries.append(entry)

            # Button frame
            btn_frame = ctk.CTkFrame(input_frame)
            btn_frame.pack(pady=30)

            # Solve button - larger
            self.solve_btn.configure(state="normal")
            solve_btn = ctk.CTkButton(
                btn_frame,
                text="Solve System",
                command=self.solve_system,
                fg_color="green",
                width=180,
                height=50,
                font=("Consolas", 16)
            )
            solve_btn.pack(side="left", padx=15)

            # New system button - larger
            ctk.CTkButton(
                btn_frame,
                text="New System",
                command=self.show_welcome_screen,
                fg_color="gray",
                width=180,
                height=50,
                font=("Consolas", 16)
            ).pack(side="left", padx=15)

            self.status_label.configure(text=f"Created {size}×{size} system", text_color="green")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create matrix: {str(e)}")

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
                    row.append(float(val))
                matrix_a.append(row)

            # Get vector b
            for i in range(size):
                val = self.b_entries[i].get()
                if val == "":
                    val = "0"
                matrix_b.append(float(val))

            return matrix_a, matrix_b

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
            return None, None

    def solve_system(self):
        matrix_a, matrix_b = self.get_matrix_values()
        if matrix_a is None:
            return

        try:
            self.status_label.configure(text="Solving...", text_color="orange")
            self.update()

            # Create solver instance
            solver = GaussJordanSolver()

            # Solve the system
            solution, rref, steps = solver.solve(matrix_a, matrix_b)

            # Store for later use
            self.solution = solution
            self.steps_data = steps
            self.matrix_a = matrix_a
            self.matrix_b = matrix_b
            self.current_step = 0

            # Show solution
            self.show_solution(solution, matrix_a, matrix_b)

            self.status_label.configure(text="Solved successfully!", text_color="green")

        except Exception as e:
            self.status_label.configure(text="Error", text_color="red")
            messagebox.showerror("Error", f"Failed to solve system: {str(e)}")

    def show_solution(self, solution, matrix_a, matrix_b):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        solution_frame = ctk.CTkFrame(self.main_frame)
        solution_frame.pack(fill="both", expand=True, padx=40, pady=40)

        # Header - larger
        header_frame = ctk.CTkFrame(solution_frame)
        header_frame.pack(fill="x", pady=15)

        ctk.CTkLabel(header_frame, text="Solution",
                     font=("Consolas", 28, "bold")).pack(side="left", padx=15)

        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back",
            command=self.create_matrix,
            fg_color="gray",
            width=120,
            height=40,
            font=("Consolas", 14)
        )
        back_btn.pack(side="right", padx=15)

        # Solution values - larger
        ctk.CTkLabel(solution_frame, text="Solution Vector X:",
                     font=("Consolas", 20)).pack(pady=25)

        size = len(solution)
        for i in range(size):
            row_frame = ctk.CTkFrame(solution_frame)
            row_frame.pack(pady=8)

            ctk.CTkLabel(row_frame, text=f"x{i + 1} = ",
                         font=("Consolas", 18)).pack(side="left", padx=8)

            # Highlight the value - larger font
            value_label = ctk.CTkLabel(
                row_frame,
                text=f"{solution[i]:.8f}",
                font=("Courier", 18, "bold"),
                text_color="green"
            )
            value_label.pack(side="left", padx=8)

        # Button frame
        btn_frame = ctk.CTkFrame(solution_frame)
        btn_frame.pack(pady=40)

        # Verify button - larger
        verify_btn = ctk.CTkButton(
            btn_frame,
            text="Verify Solution",
            command=self.verify_solution,
            fg_color="blue",
            width=180,
            height=50,
            font=("Consolas", 16)
        )
        verify_btn.pack(side="left", padx=15)

        # Steps button - larger
        self.steps_btn = ctk.CTkButton(
            btn_frame,
            text="View Step-by-Step",
            command=self.show_steps_viewer,
            fg_color="orange",
            width=180,
            height=50,
            font=("Consolas", 16)
        )
        self.steps_btn.pack(side="left", padx=15)

        # New problem button - larger
        new_btn = ctk.CTkButton(
            btn_frame,
            text="New Problem",
            command=self.clear_all,
            fg_color="green",
            width=180,
            height=50,
            font=("Consolas", 16)
        )
        new_btn.pack(side="left", padx=15)

    def verify_solution(self):
        if self.solution is None:
            return

        try:
            solver = GaussJordanSolver()
            result = solver.verify(self.matrix_a, self.matrix_b, self.solution)

            # Create verification window
            verify_window = ctk.CTkToplevel(self)
            verify_window.title("Solution Verification")
            verify_window.geometry("700x600")  # Larger window

            # Title - larger
            ctk.CTkLabel(verify_window, text="Verification Results",
                         font=("Consolas", 22, "bold")).pack(pady=25)

            # Create scrollable content
            scroll_frame = ctk.CTkScrollableFrame(verify_window)
            scroll_frame.pack(fill="both", expand=True, padx=25, pady=15)

            # Original b - larger font
            ctk.CTkLabel(scroll_frame, text="Original b:",
                         font=("Consolas", 16, "bold")).pack(anchor="w", pady=8)

            orig_text = ""
            for i, val in enumerate(result['original']):
                orig_text += f"b{i + 1} = {val:12.8f}\n"
            ctk.CTkLabel(scroll_frame, text=orig_text,
                         font=("Courier", 14), justify="left").pack(anchor="w", pady=5)

            # Calculated A*x - larger font
            ctk.CTkLabel(scroll_frame, text="Calculated A*x:",
                         font=("Consolas", 16, "bold")).pack(anchor="w", pady=(25, 8))

            calc_text = ""
            for i, val in enumerate(result['calculated']):
                calc_text += f"b{i + 1} = {val:12.8f}\n"
            ctk.CTkLabel(scroll_frame, text=calc_text,
                         font=("Courier", 14), justify="left").pack(anchor="w", pady=5)

            # Errors - larger font
            ctk.CTkLabel(scroll_frame, text="Errors (|b - A*x|):",
                         font=("Consolas", 16, "bold")).pack(anchor="w", pady=(25, 8))

            error_text = ""
            for i, val in enumerate(result['error']):
                error_text += f"Error {i + 1} = {val:12.2e}\n"
            ctk.CTkLabel(scroll_frame, text=error_text,
                         font=("Courier", 14), justify="left").pack(anchor="w", pady=5)

            # Max error - larger font
            ctk.CTkLabel(scroll_frame,
                         text=f"Maximum Error: {result['max_error']:.2e}",
                         font=("Consolas", 16, "bold")).pack(anchor="w", pady=25)

            # Conclusion - larger font
            if result['is_correct']:
                ctk.CTkLabel(scroll_frame, text="✓ Solution is CORRECT",
                             text_color="green", font=("Consolas", 18, "bold")).pack(pady=15)
            else:
                ctk.CTkLabel(scroll_frame, text="✗ Solution has ERRORS",
                             text_color="red", font=("Consolas", 18, "bold")).pack(pady=15)

            # Close button - larger
            ctk.CTkButton(verify_window, text="Close",
                          command=verify_window.destroy,
                          fg_color="gray",
                          width=120,
                          height=40,
                          font=("Consolas", 14)).pack(pady=25)

        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

    def show_steps_viewer(self):
        """Show step-by-step solution viewer with visual matrix display"""
        if not self.steps_data:
            messagebox.showinfo("No Steps", "No step-by-step data available")
            return

        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create steps viewer
        steps_frame = ctk.CTkFrame(self.main_frame)
        steps_frame.pack(fill="both", expand=True, padx=25, pady=25)

        # Header - larger
        header_frame = ctk.CTkFrame(steps_frame)
        header_frame.pack(fill="x", pady=15)

        ctk.CTkLabel(header_frame, text="Step-by-Step Solution",
                     font=("Consolas", 22, "bold")).pack(side="left", padx=15)

        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back to Solution",
            command=lambda: self.show_solution(self.solution, self.matrix_a, self.matrix_b),
            fg_color="gray",
            width=160,
            height=40,
            font=("Consolas", 14)
        )
        back_btn.pack(side="right", padx=15)

        # Navigation controls - larger
        nav_frame = ctk.CTkFrame(steps_frame)
        nav_frame.pack(fill="x", pady=15)

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

        # Main content area
        content_frame = ctk.CTkFrame(steps_frame)
        content_frame.pack(fill="both", expand=True, pady=15)

        # Left side: Step description - wider
        desc_frame = ctk.CTkFrame(content_frame, width=400)
        desc_frame.pack(side="left", fill="y", padx=(0, 20))

        self.desc_label = ctk.CTkLabel(
            desc_frame,
            text="",
            font=("Consolas", 16, "bold"),
            wraplength=380,
            justify="left"
        )
        self.desc_label.pack(pady=15, padx=15)

        # Operation details - larger textbox
        self.op_details = ctk.CTkTextbox(
            desc_frame,
            height=200,
            font=("Consolas", 14),
            wrap="word"
        )
        self.op_details.pack(fill="both", expand=True, padx=15, pady=15)
        self.op_details.configure(state="disabled")

        # Right side: Matrix display
        matrix_display_frame = ctk.CTkFrame(content_frame)
        matrix_display_frame.pack(side="right", fill="both", expand=True)

        # Matrix display area
        self.matrix_display_area = ctk.CTkFrame(matrix_display_frame)
        self.matrix_display_area.pack(fill="both", expand=True, padx=15, pady=15)

        # Show first step
        self.current_step = 0
        self.show_current_step()

    def show_current_step(self):
        """Display the current step with visual matrix"""
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

        # Add operation explanation - larger text
        desc = step['desc']
        if "Swap" in desc:
            self.op_details.insert("1.0", "OPERATION: ROW SWAPPING\n\n")
            self.op_details.insert("end", "• Swapped two rows to get a better pivot\n")
            self.op_details.insert("end", "• This improves numerical stability\n")
            self.op_details.insert("end", "• Helps avoid division by zero\n")
            self.op_details.insert("end", "• Pivot element should be the largest in column")
        elif "Normalize" in desc:
            self.op_details.insert("1.0", "OPERATION: ROW NORMALIZATION\n\n")
            self.op_details.insert("end", "• Divided the pivot row by its diagonal element\n")
            self.op_details.insert("end", "• Makes the pivot element equal to 1\n")
            self.op_details.insert("end", "• Simplifies elimination steps\n")
            self.op_details.insert("end", "• Prepare for elimination in other rows")
        elif "Eliminate" in desc:
            self.op_details.insert("1.0", "OPERATION: ROW ELIMINATION\n\n")
            self.op_details.insert("end", "• Subtracted a multiple of pivot row\n")
            self.op_details.insert("end", "• Makes other elements in pivot column zero\n")
            self.op_details.insert("end", "• Progressively creates identity matrix\n")
            self.op_details.insert("end", "• Each elimination step zeros out one element")
        else:
            self.op_details.insert("1.0", "MATRIX STATE")

        self.op_details.configure(state="disabled")

        # Display matrix
        try:
            if 'matrix' in step:
                matrix = step['matrix']
                A = matrix[:, :-1]
                b = matrix[:, -1]

                # Create visual matrix display with LARGER cells
                matrix_display = MatrixDisplay(
                    self.matrix_display_area,
                    A,
                    b,
                    title=f"Step {step['step']}: Augmented Matrix [A | b]"
                )
                matrix_display.pack(fill="both", expand=True)

                # Highlight current pivot row if applicable
                if 'Normalize' in step['desc'] or 'Eliminate' in step['desc']:
                    self.highlight_pivot_row(matrix_display, step)

            else:
                # Fallback: use A and b separately
                A = step['A']
                b = step['b']

                matrix_display = MatrixDisplay(
                    self.matrix_display_area,
                    A,
                    b,
                    title=f"Step {step['step']}: Augmented Matrix [A | b]"
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

    def highlight_pivot_row(self, matrix_display, step):
        """Highlight the current pivot row in the matrix display"""
        try:
            # Extract pivot row number from description
            desc = step['desc']
            if 'row' in desc.lower():
                # Try to find row number in description
                import re
                match = re.search(r'row\s*(\d+)', desc.lower())
                if match:
                    row_idx = int(match.group(1)) - 1  # Convert to 0-based index

                    # Highlight the row in the matrix display
                    if hasattr(matrix_display, 'cells') and row_idx < len(matrix_display.cells):
                        row_cells = matrix_display.cells[row_idx]
                        for cell in row_cells:
                            # Highlight with a different color - darker blue for better contrast
                            cell.configure(fg_color="#1a4d80", border_color="#4a90e2", border_width=2)

        except:
            pass  # If highlighting fails, just continue

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
        """CLEAR screen, data, values"""
        self.matrix_size.set(3)
        self.steps_data = []
        self.current_step = 0
        self.solution = None
        self.solve_btn.configure(state="disabled")
        self.status_label.configure(text="Ready", text_color="gray")
        self.show_welcome_screen()


if __name__ == "__main__":
    app = GaussJordanApp()
    app.mainloop()