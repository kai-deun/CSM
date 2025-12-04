'''
Author Pbjorn10
Note GUI uses a customtkinter as GUI design feature
Protoype of Main Class Guys wait lng inaaral ko pa Python
'''
import customtkinter as ctk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class FlexibleMatrixApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Flexible Matrix Generator")
        self.geometry("700x600")

        # Configure grid weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- CONTROLS FRAME ---
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # Input: ROWS
        self.lbl_rows = ctk.CTkLabel(self.control_frame, text="Rows:")
        self.lbl_rows.pack(side="left", padx=(10, 5))
        self.entry_rows = ctk.CTkEntry(self.control_frame, width=40, placeholder_text="2")
        self.entry_rows.pack(side="left", padx=5)

        # Input: COLUMNS
        self.lbl_cols = ctk.CTkLabel(self.control_frame, text="Cols:")
        self.lbl_cols.pack(side="left", padx=(10, 5))
        self.entry_cols = ctk.CTkEntry(self.control_frame, width=40, placeholder_text="3")
        self.entry_cols.pack(side="left", padx=5)

        # Button: Generate
        self.btn_gen = ctk.CTkButton(self.control_frame, text="Create Grid", command=self.generate_grid)
        self.btn_gen.pack(side="left", padx=20)

        # --- THE MATRIX BOX ---
        self.matrix_box = ctk.CTkScrollableFrame(self, label_text="Matrix Input Area")
        self.matrix_box.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # --- ACTION BUTTON ---
        self.btn_print = ctk.CTkButton(self, text="Print Values to Console", command=self.get_values, fg_color="green")
        self.btn_print.grid(row=2, column=0, padx=20, pady=20)

        self.matrix_entries = []  # Storage for our text fields

    def generate_grid(self):
        # 1. Get Rows and Cols inputs
        try:
            rows = int(self.entry_rows.get())
            cols = int(self.entry_cols.get())
        except ValueError:
            return  # Stop if inputs are empty or not numbers

        # 2. Clear old grid (Like panel.removeAll() in Java)
        for widget in self.matrix_box.winfo_children():
            widget.destroy()
        self.matrix_entries = []

        # 3. The Nested Loop (O(rows * cols))
        # Logic: We build it row by row.
        for r in range(rows):
            current_row_widgets = []  # Temp list for this row

            for c in range(cols):
                # Determine placeholder text (e.g., "0,0", "0,1")
                # This helps the user know which coordinate they are typing in
                placeholder = f"({r},{c})"

                # Check if it's the last column (Optional: Make it look like the "Answer" column)
                if c == cols - 1:
                    color = "#3a3a3a"  # Slightly darker for the last column
                else:
                    color = None  # Default color

                entry = ctk.CTkEntry(self.matrix_box, width=60, placeholder_text=placeholder, fg_color=color)

                # Grid logic: simple r, c mapping
                entry.grid(row=r, column=c, padx=3, pady=3)

                current_row_widgets.append(entry)

            # Add the row of widgets to our main list
            self.matrix_entries.append(current_row_widgets)

    def get_values(self):
        # Extract data
        data = []
        try:
            for r_entries in self.matrix_entries:
                row_values = []
                for entry in r_entries:
                    val = entry.get()
                    if val == "": val = "0"  # Default to 0 if empty
                    row_values.append(float(val))
                data.append(row_values)

            # Print cleanly
            print(f"Captured {len(data)}x{len(data[0])} Matrix:")
            for row in data:
                print(row)

        except ValueError:
            print("Error: Matrix contains non-numeric values.")


if __name__ == "__main__":
    app = FlexibleMatrixApp()
    app.mainloop()