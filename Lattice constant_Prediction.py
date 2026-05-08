import os
import sys
import threading
import queue
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ase import Atoms
from ase.build import bulk, molecule
from ase.data import chemical_symbols
from ase.collections import g2

import engine

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# ─── Compatibility Matrix ────────────────────────────────────────
# Defines which simulation modes make physical sense for each material type.
ALL_MODES = ["Equilibrium Scan", "Equation of State", "Relaxation", "Tensile Test",
             "Phonon", "Molecular Dynamics", "Diffusion", "Phase Diagram", "Thermodynamics", "Vapor Pressure"]

MODE_COMPATIBILITY = {
    "element": ALL_MODES,           # Crystal elements: all modes valid
    "molecule": [
        # Molecules are not periodic crystals — no lattice constant, no phase diagram, no tensile
        "Relaxation", "Phonon", "Molecular Dynamics", "Diffusion", "Thermodynamics", "Vapor Pressure"
    ],
    "compound": [
        # Compounds (e.g. NaCl): EOS, relaxation, thermodynamics valid
        "Equation of State", "Relaxation", "Thermodynamics", "Vapor Pressure"
    ],
}

# Tooltip explanations for disabled modes
MODE_DISABLE_REASON = {
    "Equilibrium Scan":   "Requires a periodic crystal (Element only)",
    "Tensile Test":       "Requires periodic crystal with known structure",
    "Phase Diagram":      "Requires 2 crystal phases of same element",
    "Phonon":             "Requires crystal periodicity (not available for Compound)",
    "Molecular Dynamics": "Requires crystal periodicity (not available for Compound)",
    "Diffusion":          "Requires crystal periodicity (not available for Compound)",
}
# ────────────────────────────────────────────────────────────────

ELEMENT_STRUCTURES = {
    # Metals (Common phases)
    "Fe": ["bcc", "fcc", "hcp"],
    "Al": ["fcc"],
    "Na": ["bcc", "fcc"],
    "Mg": ["hcp"],
    "Cu": ["fcc"],
    "Ni": ["fcc"],
    "Ag": ["fcc"],
    "Au": ["fcc"],
    "Ti": ["hcp", "bcc"],
    "Zn": ["hcp"],
    "Cr": ["bcc"],
    "Mo": ["bcc"],
    "W":  ["bcc"],
    "Pb": ["fcc"],
    "Li": ["bcc", "fcc"],
    "K":  ["bcc"],
    "Ca": ["fcc", "bcc"],
    "Pt": ["fcc"],
    "Pd": ["fcc"],
    "Co": ["hcp", "fcc"],
    "V":  ["bcc"],
    "Nb": ["bcc"],
    "Ta": ["bcc"],
    "Be": ["hcp"],
    "Cd": ["hcp"],
    
    # Semi-metals / Non-metals (Crystalline)
    "C":  ["diamond", "sc"],
    "Si": ["diamond"],
    "Ge": ["diamond"],
    "Sn": ["fcc", "diamond"],
}

# Elements that should be handled as Molecules, not Bulk Crystals
MOLECULAR_ELEMENTS = ["H", "He", "N", "O", "F", "Ne", "Cl", "Ar", "Kr", "Xe", "Br", "I"]

# Common stable molecules from G2 collection
COMMON_MOLECULES = [
    'H2', 'N2', 'O2', 'F2', 'Cl2', 'H2O', 'CO', 'CO2', 'NH3', 'CH4',
    'C2H2', 'C2H4', 'C2H6', 'C3H8', 'C6H6', 'CH3OH', 'CH3CH2OH',
    'CH3CN', 'CH3Cl', 'CCl4', 'CF4', 'HCN', 'HCl', 'HF', 'HBr', 'HI',
    'SO2', 'H2S', 'H2O2', 'N2O', 'NO', 'NO2', 'BF3', 'AlCl3', 'SiH4',
    'PH3', 'NaCl', 'LiF', 'LiH', 'O3', 'CS2', 'CH3COCH3', 'CH3CHO'
]

DEFAULT_STRUCTURES = [] # No default for unknown elements to prevent fake results

class SearchableComboBox(ctk.CTkFrame):
    def __init__(self, master, values, width=140, command=None, **kwargs):
        super().__init__(master, width=width, fg_color="transparent")
        self._all_values = list(values)
        self._command = command
        self._width = width
        self._popup = None
        self._is_selecting = False # Flag to prevent feedback loop

        self._var = ctk.StringVar()
        self._entry = ctk.CTkEntry(self, textvariable=self._var, width=width - 30)
        self._entry.pack(side=tk.LEFT)

        self._btn = ctk.CTkButton(self, text="▼", width=28, height=28,
                                   command=self._toggle_popup, fg_color="gray30",
                                   hover_color="gray40")
        self._btn.pack(side=tk.LEFT, padx=(1, 0))

        self._var.trace_add("write", self._on_type)
        self._entry.bind("<FocusIn>", lambda e: self._show_popup())
        self._entry.bind("<Return>", lambda e: self._select_top())
        
        # Binding FocusOut cho Entry va Button
        self._entry.bind("<FocusOut>", self._on_focus_out)
        self._btn.bind("<FocusOut>", self._on_focus_out)

    def _on_type(self, *args):
        if self._is_selecting: return
        self._show_popup()

    def _show_popup(self):
        self.update_idletasks()
        val = self._var.get().strip().lower()
        filtered = [v for v in self._all_values if v.lower().startswith(val)]
        
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
        if not filtered: return
        
        try:
            x = self._entry.winfo_rootx()
            y = self._entry.winfo_rooty() + self._entry.winfo_height() + 2
        except: return

        h = min(220, len(filtered) * 26)
        self._popup = tk.Toplevel(self)
        self._popup.wm_overrideredirect(True)
        self._popup.geometry(f"{self._width}x{h}+{x}+{y}")
        self._popup.configure(bg="#2b2b2b")
        self._popup.attributes("-topmost", True)
        
        self._scrollbar = tk.Scrollbar(self._popup, orient=tk.VERTICAL, bg="#333")
        self._listbox = tk.Listbox(
            self._popup, bg="#2b2b2b", fg="#dde",
            selectbackground="#1f6aa5", relief=tk.FLAT, bd=0,
            yscrollcommand=self._scrollbar.set
        )
        self._scrollbar.config(command=self._listbox.yview)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(fill=tk.BOTH, expand=True)
        
        for v in filtered:
            self._listbox.insert(tk.END, v)
        
        # Click selection logic - Instant selection
        self._listbox.bind("<<ListboxSelect>>", lambda e: self._on_select())
        
        # Bat FocusOut tren ca Popup va Listbox
        self._popup.bind("<FocusOut>", self._on_focus_out)
        self._listbox.bind("<FocusOut>", self._on_focus_out)

    def _on_list_click(self, event):
        idx = self._listbox.nearest(event.y)
        if idx >= 0:
            val = self._listbox.get(idx)
            self._var.set(val)
            if self._command: self._command(val)
        self._close_popup()

    def _on_focus_out(self, event=None):
        """Wait 10ms to see where the focus goes."""
        self.after(10, self._check_focus)

    def _check_focus(self):
        if not self._popup or not self._popup.winfo_exists():
            return
        
        # Get the current widget with focus
        focused = self.focus_get()
        
        # Safe widgets belonging to this ComboBox
        safe_widgets = (
            self._entry, 
            self._btn, 
            self._popup, 
            getattr(self, '_listbox', None), 
            getattr(self, '_scrollbar', None)
        )
        
        # If focus moves outside, close the popup
        if focused not in safe_widgets:
            self._close_popup()

    def _on_select(self):
        sel = self._listbox.curselection()
        if sel:
            val = self._listbox.get(sel[0])
            self._is_selecting = True
            self._var.set(val)
            self._is_selecting = False
            if self._command: self._command(val)
        self._close_popup()
        self.master.focus_set() # Move focus out to confirm selection

    def _select_top(self):
        query = self._var.get().strip().lower()
        filtered = [v for v in self._all_values if v.lower().startswith(query)]
        if filtered:
            val = filtered[0]
            self._var.set(val)
            if self._command: self._command(val)
        self._close_popup()

    def _toggle_popup(self):
        if self._popup and self._popup.winfo_exists():
            self._close_popup()
        else:
            self._entry.focus_set()
            self._show_popup()

    def _close_popup(self):
        if self._popup:
            try: self._popup.destroy()
            except: pass
            self._popup = None

    def _focus_list(self):
        if self._popup and self._popup.winfo_exists():
            self._listbox.focus_set()
            self._listbox.selection_set(0)

    def get(self):
        return self._var.get()

    def set(self, value):
        self._var.set(value)

    def configure(self, **kwargs):
        state = kwargs.pop("state", None)
        if state:
            if state == "disabled":
                self._entry.configure(state="disabled", text_color="gray50")
                self._btn.configure(state="disabled", fg_color="gray20", text_color="gray40")
            else:
                self._entry.configure(state="normal", text_color=("gray10", "gray90"))
                self._btn.configure(state="normal", fg_color="gray30", text_color="white")
        if kwargs:
            super().configure(**kwargs)

    def cget(self, attr):
        if attr == "state":
            return self._entry.cget("state")
        return super().cget(attr)


class MatterSimApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MatterSim AI Materials Lab (Refactored)")
        self.geometry("1280x900")
        
        self.task_queue = queue.Queue()
        self.is_running = False
        
        # Variables
        self.selection_type = ctk.StringVar(value="element")
        self.md_engine = ctk.StringVar(value="ase")
        self.num_gpus = ctk.IntVar(value=1)
        self.verbose = ctk.BooleanVar(value=True)
        
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._safe_exit)
        self.after(100, self._process_queue)

    def _build_ui(self):
        # Create a scrollable frame for the main content
        self.main_frame = ctk.CTkScrollableFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Settings Panel ---
        self.settings_frame = ctk.CTkFrame(self.main_frame)
        self.settings_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ctk.CTkLabel(self.settings_frame, text="Computation Settings", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(self.settings_frame, text="GPUs:").grid(row=0, column=1, padx=5)
        self.gpu_entry = ctk.CTkEntry(self.settings_frame, width=50)
        self.gpu_entry.insert(0, "1")
        self.gpu_entry.grid(row=0, column=2, padx=5)
        
        self.verbose_check = ctk.CTkCheckBox(self.settings_frame, text="Verbose Logging", variable=self.verbose)
        self.verbose_check.grid(row=0, column=3, padx=20)
        
        # --- Material Selection Panel ---
        self.sel_frame = ctk.CTkFrame(self.main_frame)
        self.sel_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ctk.CTkLabel(self.sel_frame, text="Material Selection", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.radio_element = ctk.CTkRadioButton(self.sel_frame, text="Element", variable=self.selection_type, value="element", command=self._update_selection_widgets)
        self.radio_element.grid(row=1, column=0, padx=10, pady=5)
        
        self.radio_molecule = ctk.CTkRadioButton(self.sel_frame, text="Molecule", variable=self.selection_type, value="molecule", command=self._update_selection_widgets)
        self.radio_molecule.grid(row=1, column=1, padx=10, pady=5)
        
        self.radio_compound = ctk.CTkRadioButton(self.sel_frame, text="Compound", variable=self.selection_type, value="compound", command=self._update_selection_widgets)
        self.radio_compound.grid(row=1, column=2, padx=10, pady=5)
        
        # Element inputs — searchable (Only solid elements with defined crystals)
        self.element_cb = SearchableComboBox(
            self.sel_frame,
            values=sorted(list(ELEMENT_STRUCTURES.keys())),
            width=140,
            command=self._update_structure_options
        )
        self.element_cb.set("Fe")
        self.element_cb.grid(row=2, column=0, padx=10, pady=5)

        # Element structures — dynamic list
        self.structure_cb = ctk.CTkOptionMenu(self.sel_frame, values=["bcc", "fcc", "hcp"])
        self.structure_cb.set("bcc")
        self.structure_cb.grid(row=2, column=1, padx=10, pady=5)

        # Orientation — Only relevant for crystals in Tensile Test
        ctk.CTkLabel(self.sel_frame, text="Orient:").grid(row=2, column=2, padx=5)
        self.orient_cb = ctk.CTkOptionMenu(self.sel_frame, values=["[001]", "[111]"], width=80)
        self.orient_cb.set("[001]")
        self.orient_cb.grid(row=2, column=3, padx=10, pady=5)

        # Molecule inputs — searchable (Common stable molecules only)
        self.molecule_cb = SearchableComboBox(
            self.sel_frame,
            values=sorted(COMMON_MOLECULES),
            width=140
        )
        self.molecule_cb.set("H2O")
        self.molecule_cb.grid(row=3, column=0, padx=10, pady=5)
        
        # Compound inputs
        self.formula_entry = ctk.CTkEntry(self.sel_frame, placeholder_text="Formula (e.g. NaCl)")
        self.formula_entry.grid(row=4, column=0, padx=10, pady=5)
        
        # Calculation Mode
        ctk.CTkLabel(self.sel_frame, text="Calculation Mode:").grid(row=3, column=2, padx=10)
        self.mode_cb = ctk.CTkOptionMenu(self.sel_frame, values=[
            "Equilibrium Scan", "Equation of State", "Relaxation", "Tensile Test", "Phonon", "Molecular Dynamics", "Diffusion", "Phase Diagram", "Thermodynamics", "Vapor Pressure"
        ], command=self._on_mode_change)
        self.mode_cb.set("Equilibrium Scan")
        self.mode_cb.grid(row=3, column=3, padx=10, pady=5)
        
        self.run_btn = ctk.CTkButton(self.sel_frame, text="▶ Run Simulation", command=self._on_run, fg_color="green", hover_color="darkgreen")
        self.run_btn.grid(row=4, column=3, padx=10, pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.sel_frame, mode="indeterminate")
        self.progress_bar.grid(row=5, column=3, padx=10, pady=5, sticky="ew")
        self.progress_bar.set(0)
        
        # --- Output Tabs ---
        self.tabview = ctk.CTkTabview(self.main_frame, width=1200, height=600)
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tabs = {}
        for name in ["Equilibrium Scan", "Equation of State", "Relaxation", "Tensile Test", "Phonon", "Molecular Dynamics", "Diffusion", "Phase Diagram", "Thermodynamics", "Vapor Pressure"]:
            self.tabs[name] = self.tabview.add(name)
            
        self._build_equilibrium_tab()
        self._build_eos_tab()
        self._build_relaxation_tab()
        self._build_tensile_tab()
        self._build_phonon_tab()
        self._build_md_tab()
        self._build_diffusion_tab()
        self._build_phase_tab()
        self._build_thermo_tab()
        self._build_vapor_pressure_tab()
        
        # Sync tab clicks with mode selection
        try:
            self.tabview._segmented_button.configure(command=self._on_tab_change)
        except: pass

        self._update_selection_widgets()
        
    def _update_selection_widgets(self):
        sel = self.selection_type.get()
        mode = self.mode_cb.get()
        
        # Close any open popups when switching
        self.element_cb._close_popup()
        self.molecule_cb._close_popup()
        
        # 1. Material Selection widgets
        self.element_cb.configure(state="normal" if sel == "element" else "disabled")
        self.molecule_cb.configure(state="normal" if sel == "molecule" else "disabled")
        self.formula_entry.configure(state="normal" if sel == "compound" else "disabled")
        
        # 2. Structure Dropdown - ABSOLUTE ENFORCEMENT
        if mode == "Phase Diagram":
            self.structure_cb.configure(state="disabled")
        else:
            self.structure_cb.configure(state="normal" if sel == "element" else "disabled")

        # 3. Orientation - ONLY for Tensile Test of Elements
        if mode == "Tensile Test" and sel == "element":
            self.orient_cb.configure(state="normal")
        else:
            self.orient_cb.configure(state="disabled")
            
        self._update_available_modes(sel)

    def _update_available_modes(self, selection):
        """Dynamic UI filtering: Hide incompatible tabs for Element, Molecule, and Compound."""
        # Use the predefined Compatibility Matrix
        active_modes = MODE_COMPATIBILITY.get(selection, ALL_MODES)
        
        # 1. Update Dropdown
        self.mode_cb.configure(values=active_modes)
        
        # 2. Update Tabview (Sync with dropdown)
        mode = self.mode_cb.get()
        try:
            self.tabview.set(mode)
        except Exception as e:
            print(f"[UI Warning] Could not set tab to '{mode}': {e}")
        # 3. Fallback: If current mode is not in the active set, switch to a valid one
        current = self.mode_cb.get()
        if current not in active_modes:
            # Try Relaxation first as a stable fallback, otherwise take first available
            fallback = "Relaxation" if "Relaxation" in active_modes else active_modes[0]
            self.mode_cb.set(fallback)
            self.tabview.set(fallback)
            self._update_selection_widgets()

    def _on_mode_change(self, choice):
        """Unified entry point for calculation mode change (via Dropdown)."""
        print(f"[UI] Mode Dropdown changed to: '{choice}'")
        # 1. Ép chuyển Tab nội dung
        self.tabview.set(choice)
        # 2. Cập nhật các widget liên quan
        self._update_selection_widgets()

    def _on_tab_change(self, tab_name):
        """Unified entry point for calculation mode change (via Tab click)."""
        selection = self.selection_type.get()
        valid_modes = MODE_COMPATIBILITY.get(selection, ALL_MODES)
        
        # BLOCK access if the mode is not supported for this material type
        if tab_name not in valid_modes:
            # Revert tab selection to current mode
            self.tabview.set(self.mode_cb.get())
            return
            
        # Synchronize dropdown
        self.mode_cb.set(tab_name)
        self._update_selection_widgets()

    def _update_structure_options(self, element):
        """Update structure dropdown based on selected element."""
        # Only solid elements are selectable now, so we just lookup the structure list
        valid_structs = ELEMENT_STRUCTURES.get(element, ["bcc", "fcc", "hcp"])
        
        mode = self.mode_cb.get()
        state = "disabled" if mode == "Phase Diagram" else "normal"
        self.structure_cb.configure(values=valid_structs, state=state)
        
        if self.structure_cb.get() not in valid_structs:
            self.structure_cb.set(valid_structs[0])

        # Update Phase Diagram dropdowns
        self.pg_phase1_cb.configure(values=valid_structs, state="normal")
        self.pg_phase2_cb.configure(values=valid_structs, state="normal")
        if self.pg_phase1_cb.get() not in valid_structs:
            self.pg_phase1_cb.set(valid_structs[0])
        if self.pg_phase2_cb.get() not in valid_structs:
            self.pg_phase2_cb.set(valid_structs[min(1, len(valid_structs)-1)])

    # --- Tab Builders ---
    def _build_equilibrium_tab(self):
        frm = self.tabs["Equilibrium Scan"]
        self.equi_input_frame = ctk.CTkFrame(frm)
        self.equi_input_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(self.equi_input_frame, text="± Scan around guess (%):").pack(side=tk.LEFT, padx=5)
        self.equi_pct = ctk.CTkEntry(self.equi_input_frame, width=60)
        self.equi_pct.insert(0, "10")
        self.equi_pct.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkLabel(self.equi_input_frame, text="Number of points:").pack(side=tk.LEFT, padx=5)
        self.equi_npts = ctk.CTkEntry(self.equi_input_frame, width=60)
        self.equi_npts.insert(0, "20")
        self.equi_npts.pack(side=tk.LEFT, padx=5)
        
        # Result summary label
        self.equi_res_label = ctk.CTkLabel(frm, text="Results: Ready", font=ctk.CTkFont(weight="bold", size=14), text_color="#2ecc71")
        self.equi_res_label.pack(pady=2)
        
        self.equi_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.equi_ax = self.equi_fig.add_subplot(111)
        self.equi_canvas = FigureCanvasTkAgg(self.equi_fig, master=frm)
        self.equi_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_eos_tab(self):
        frm = self.tabs["Equation of State"]
        # Result summary label
        self.eos_res_label = ctk.CTkLabel(frm, text="Results: Ready", font=ctk.CTkFont(weight="bold", size=14), text_color="#3498db")
        self.eos_res_label.pack(pady=5)
        
        self.eos_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.eos_ax = self.eos_fig.add_subplot(111)
        self.eos_canvas = FigureCanvasTkAgg(self.eos_fig, master=frm)
        self.eos_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_tensile_tab(self):
        frm = self.tabs["Tensile Test"]
        # Result summary label
        self.tensile_res_label = ctk.CTkLabel(frm, text="Results: Ready", font=ctk.CTkFont(weight="bold", size=14), text_color="#e67e22")
        self.tensile_res_label.pack(pady=5)
        
        self.tensile_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.tensile_ax = self.tensile_fig.add_subplot(111)
        self.tensile_canvas = FigureCanvasTkAgg(self.tensile_fig, master=frm)
        self.tensile_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_relaxation_tab(self):
        frm = self.tabs["Relaxation"]
        self.relax_log = ctk.CTkTextbox(frm, height=400)
        self.relax_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_phonon_tab(self):
        frm = self.tabs["Phonon"]
        self.phonon_frame = ctk.CTkFrame(frm)
        self.phonon_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_md_tab(self):
        frm = self.tabs["Molecular Dynamics"]
        input_frm = ctk.CTkFrame(frm)
        input_frm.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(input_frm, text="Temperature (K):").pack(side=tk.LEFT, padx=5)
        self.md_temp = ctk.CTkEntry(input_frm, width=60)
        self.md_temp.insert(0, "300")
        self.md_temp.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkLabel(input_frm, text="Steps:").pack(side=tk.LEFT, padx=5)
        self.md_steps = ctk.CTkEntry(input_frm, width=60)
        self.md_steps.insert(0, "100")
        self.md_steps.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkLabel(input_frm, text="Ensemble:").pack(side=tk.LEFT, padx=5)
        self.md_ensemble = ctk.CTkOptionMenu(input_frm, values=["NVT", "NPT"], width=80)
        self.md_ensemble.set("NVT")
        self.md_ensemble.pack(side=tk.LEFT, padx=5)

        ctk.CTkLabel(input_frm, text="Pressure (GPa):").pack(side=tk.LEFT, padx=5)
        self.md_pressure = ctk.CTkEntry(input_frm, width=60)
        self.md_pressure.insert(0, "0.0")
        self.md_pressure.pack(side=tk.LEFT, padx=5)

        # Result summary label
        self.md_res_label = ctk.CTkLabel(frm, text="Status: Ready", font=ctk.CTkFont(weight="bold", size=13))
        self.md_res_label.pack(pady=2)
        
        self.md_fig = plt.Figure(figsize=(6, 4))
        self.md_ax = self.md_fig.add_subplot(111)
        self.md_canvas = FigureCanvasTkAgg(self.md_fig, master=frm)
        self.md_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_diffusion_tab(self):
        frm = self.tabs["Diffusion"]
        input_frm = ctk.CTkFrame(frm)
        input_frm.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(input_frm, text="Temperature (K):").pack(side=tk.LEFT, padx=5)
        self.diff_temp = ctk.CTkEntry(input_frm, width=60)
        self.diff_temp.insert(0, "1000")
        self.diff_temp.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkLabel(input_frm, text="Steps:").pack(side=tk.LEFT, padx=5)
        self.diff_steps = ctk.CTkEntry(input_frm, width=60)
        self.diff_steps.insert(0, "500")
        self.diff_steps.pack(side=tk.LEFT, padx=5)
        
        # Result summary label
        self.diff_res_label = ctk.CTkLabel(frm, text="Results: Ready", font=ctk.CTkFont(weight="bold", size=14), text_color="#9b59b6")
        self.diff_res_label.pack(pady=5)
        
        self.diff_fig = plt.Figure(figsize=(6, 4))
        self.diff_ax = self.diff_fig.add_subplot(111)
        self.diff_canvas = FigureCanvasTkAgg(self.diff_fig, master=frm)
        self.diff_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_phase_tab(self):
        frm = self.tabs["Phase Diagram"]
        self.phase_input_frame = ctk.CTkFrame(frm)
        self.phase_input_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(self.phase_input_frame, text="Phase 1:").grid(row=0, column=0, padx=5)
        self.pg_phase1_cb = ctk.CTkOptionMenu(self.phase_input_frame, values=["bcc", "fcc", "hcp", "diamond", "sc"])
        self.pg_phase1_cb.set("bcc")
        self.pg_phase1_cb.grid(row=0, column=1, padx=5)
        
        ctk.CTkLabel(self.phase_input_frame, text="Phase 2:").grid(row=0, column=2, padx=5)
        self.pg_phase2_cb = ctk.CTkOptionMenu(self.phase_input_frame, values=["bcc", "fcc", "hcp", "diamond", "sc"])
        self.pg_phase2_cb.set("fcc")
        self.pg_phase2_cb.grid(row=0, column=3, padx=5)
        
        ctk.CTkLabel(self.phase_input_frame, text="T Min (K):").grid(row=1, column=0, padx=5, pady=5)
        self.pg_tmin = ctk.CTkEntry(self.phase_input_frame, width=60)
        self.pg_tmin.insert(0, "0")
        self.pg_tmin.grid(row=1, column=1, padx=5)
        
        ctk.CTkLabel(self.phase_input_frame, text="T Max (K):").grid(row=1, column=2, padx=5, pady=5)
        self.pg_tmax = ctk.CTkEntry(self.phase_input_frame, width=60)
        self.pg_tmax.insert(0, "2000")
        self.pg_tmax.grid(row=1, column=3, padx=5)
        
        self.phase_fig = plt.Figure(figsize=(6, 4))
        self.phase_ax = self.phase_fig.add_subplot(111)
        self.phase_canvas = FigureCanvasTkAgg(self.phase_fig, master=frm)
        self.phase_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_thermo_tab(self):
        frm = self.tabs["Thermodynamics"]
        self.thermo_fig = plt.Figure(figsize=(8, 6))
        self.thermo_ax = self.thermo_fig.add_subplot(111)
        self.thermo_canvas = FigureCanvasTkAgg(self.thermo_fig, master=frm)
        self.thermo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_vapor_pressure_tab(self):
        frm = self.tabs["Vapor Pressure"]
        input_frm = ctk.CTkFrame(frm)
        input_frm.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(input_frm, text="T min (K):").pack(side=tk.LEFT, padx=5)
        self.vp_tmin = ctk.CTkEntry(input_frm, width=60)
        self.vp_tmin.insert(0, "300")
        self.vp_tmin.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkLabel(input_frm, text="T max (K):").pack(side=tk.LEFT, padx=5)
        self.vp_tmax = ctk.CTkEntry(input_frm, width=60)
        self.vp_tmax.insert(0, "2000")
        self.vp_tmax.pack(side=tk.LEFT, padx=5)
        
        self.vp_res_label = ctk.CTkLabel(frm, text="Cohesive Energy: ---", font=ctk.CTkFont(weight="bold", size=14), text_color="#1abc9c")
        self.vp_res_label.pack(pady=5)
        
        self.vp_fig = plt.Figure(figsize=(6, 4))
        self.vp_ax = self.vp_fig.add_subplot(111)
        self.vp_canvas = FigureCanvasTkAgg(self.vp_fig, master=frm)
        self.vp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_compound(self, formula):
        """
        Build a proper crystal structure for a compound formula.
        Uses a lookup table for common compounds, falls back to Atoms() for unknown ones.
        """
        # Lookup table: formula -> (structure, lattice_a)
        COMPOUND_CRYSTALS = {
            "nacl":  ("rocksalt",  5.64,  "NaCl"),
            "kcl":   ("rocksalt",  6.29,  "KCl"),
            "mgo":   ("rocksalt",  4.21,  "MgO"),
            "cao":   ("rocksalt",  4.80,  "CaO"),
            "tio2":  None,   # rutile — complex, fallback to Atoms
            "fe2o3": None,   # corundum — complex, fallback to Atoms
            "al2o3": None,
            "sic":   ("zincblende", 4.36, "SiC"),
            "gaas":  ("zincblende", 5.65, "GaAs"),
            "gap":   ("zincblende", 5.45, "GaP"),
            "inp":   ("zincblende", 5.87, "InP"),
        }
        key = formula.lower().replace(" ", "")
        if key in COMPOUND_CRYSTALS and COMPOUND_CRYSTALS[key] is not None:
            struct, a, sym = COMPOUND_CRYSTALS[key]
            return bulk(sym, struct, a=a)
        else:
            # Fallback: Create a simple cubic box and spread atoms to avoid overlap
            from ase.data import chemical_symbols
            import re
            # Extract symbols and counts (e.g., Fe2O3 -> [Fe, Fe, O, O, O])
            parts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
            atoms_list = []
            for sym, count in parts:
                n = int(count) if count else 1
                atoms_list.extend([sym] * n)
            
            if not atoms_list: return Atoms(formula)
            
            # Create a box large enough (e.g., 5A side)
            atoms = Atoms(atoms_list)
            n_atoms = len(atoms)
            side = 3.0 + n_atoms**0.33 * 2.0
            atoms.set_cell([side, side, side])
            atoms.set_pbc(True)
            # Randomize positions slightly to avoid (0,0,0) singularity
            pos = np.random.rand(n_atoms, 3) * side
            atoms.set_positions(pos)
            return atoms

    # --- Run Logic & Thread Management ---
    def _on_run(self):
        if self.is_running:
            return
            
        # SINGLE SOURCE OF TRUTH: Always take mode from the calculation mode dropdown
        mode = self.mode_cb.get()
        # Synchronize tabview just in case it was out of sync
        self.tabview.set(mode)
        
        selection = self.selection_type.get()
        
        # Guard: check if current tab/mode is compatible with selected material type
        valid_modes = MODE_COMPATIBILITY.get(selection, ALL_MODES)
        if mode not in valid_modes:
            reason = MODE_DISABLE_REASON.get(mode, "Not supported for this material type.")
            messagebox.showwarning(
                "Incompatible Mode",
                f"'{mode}' is not available for {selection.title()}.\n\n"
                f"Reason: {reason}\n\n"
                f"Available modes: {', '.join(valid_modes)}"
            )
            return
        
        try:
            # Input validation and Atoms building
            if selection == "element":
                element = self.element_cb.get()
                phase = self.structure_cb.get()
                use_cubic = (phase in ['bcc', 'fcc'])
                try:
                    atoms = bulk(element, phase, cubic=use_cubic)
                except Exception:
                    # Cải thiện giá trị cứu hộ: HCP thường có a nhỏ hơn FCC/BCC
                    a_guess = 2.5 if phase == 'hcp' else 3.5
                    atoms = bulk(element, phase, a=a_guess, cubic=use_cubic)
                formula = element
            elif selection == "molecule":
                molecule_name = self.molecule_cb.get()
                atoms = molecule(molecule_name)
                element = molecule_name
                phase = "molecule"
                formula = molecule_name
            else:
                formula = self.formula_entry.get().strip()
                if not formula:
                    raise ValueError("Formula cannot be empty.")
                try:
                    atoms = self._build_compound(formula)
                except Exception as e:
                    raise ValueError(f"Cannot build '{formula}': {e}")
                element = formula
                phase = "compound"
                
            # Prepare structure (standardize for bulk vs molecule)
            is_mol = (selection == "molecule")
            atoms = engine.prepare_structure(atoms, is_molecule=is_mol)
                
            self.tabview.set(mode)
            self.run_btn.configure(state="disabled", text="Running...")
            self.progress_bar.start()
            self.is_running = True
            
            # Start background thread
            threading.Thread(target=self._run_calculation_thread, args=(mode, atoms, element, phase, formula, is_mol), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Validation Error", f"Input error: {e}")
            self.run_btn.configure(state="normal", text="▶ Run Simulation")
            self.is_running = False

    def _run_calculation_thread(self, mode, atoms, element, phase, formula, is_mol):
        try:
            result = {"mode": mode, "element": element, "phase": phase, "formula": formula}
            
            if mode == "Equilibrium Scan":
                pct = float(self.equi_pct.get()) / 100.0
                npts = int(self.equi_npts.get())
                a_vals, energies = engine.run_equilibrium_scan(atoms, pct, npts)
                result["a_vals"] = a_vals
                result["energies"] = energies
                
            elif mode == "Equation of State":
                volumes, energies, eos, v0, B_GPa = engine.run_equation_of_state(atoms, pct_range=0.04, npts=15)
                result["volumes"] = volumes
                result["energies"] = energies
                result["eos"] = eos
                result["v0"] = v0
                result["B_GPa"] = B_GPa
                
            elif mode == "Tensile Test":
                orientation = self.orient_cb.get()
                strains, stresses, avg_nu = engine.run_tensile_test(atoms, strain_max=0.10, steps=10, is_molecule=is_mol, orientation=orientation, phase=phase)
                result["strains"] = strains
                result["stresses"] = stresses
                result["avg_nu"] = avg_nu

            elif mode == "Relaxation":
                from datetime import datetime
                relax_results = engine.run_relaxation(atoms, verbose=self.verbose.get(), is_molecule=is_mol)
                result["relax_results"] = relax_results
                
                # CẬP NHẬT CẤU TRÚC TỐI ƯU VÀO BỘ NHỚ CHÍNH
                self.atoms = relax_results.get('final_atoms', atoms)
                
                # --- NHẬT KÝ HÀNH TRÌNH DỮ LIỆU ---
                print("\n" + "✅"*3 + " TRẠNG THÁI BÀN GIAO BIẾN " + "✅"*3)
                print(f"DEBUG: Relaxation hoàn tất lúc {datetime.now().strftime('%H:%M:%S')}")
                print(f"DEBUG: Biến 'self.atoms' đã được cập nhật thành công.")
                fmax_val = relax_results.get('fmax', 0.0)
                print(f"DEBUG: Max Force hiện tại: {fmax_val:.6f} eV/Å")
                if len(self.atoms) > 0:
                    print(f"DEBUG: Tọa độ nguyên tử đầu tiên: {self.atoms.positions[0]}")
                print("TERMINAL: Hệ thống đã sẵn sàng truyền dữ liệu sang chế độ Phonon.")
                print("="*40 + "\n")
                
            elif mode == "Phonon":
                work_dir = f"./phonon_{element}_{phase}"
                has_imag, fig, plot_path = engine.run_phonon(atoms, work_dir, is_molecule=is_mol)
                result["has_imag"] = has_imag
                result["fig"] = fig
                result["plot_path"] = plot_path
                
            elif mode == "Molecular Dynamics":
                T = float(self.md_temp.get())
                steps = int(self.md_steps.get())
                num_gpus = int(self.gpu_entry.get())
                ensemble = self.md_ensemble.get()
                pressure = float(self.md_pressure.get())
                times, energies, temps, volumes = engine.run_molecular_dynamics(atoms, T, steps, num_gpus, ensemble, pressure)
                result["times"] = times
                result["energies"] = energies
                result["temps"] = temps
                result["volumes"] = volumes
                result["ensemble"] = ensemble
                result["pressure"] = pressure
                
            elif mode == "Diffusion":
                T = float(self.diff_temp.get())
                steps = int(self.diff_steps.get())
                num_gpus = int(self.gpu_entry.get())
                times, msd_vals, D = engine.run_diffusion(atoms, T, steps, num_gpus)
                result["times"] = times
                result["msd_vals"] = msd_vals
                result["D"] = D

            elif mode == "Phase Diagram":
                phase1 = self.pg_phase1_cb.get()
                phase2 = self.pg_phase2_cb.get()
                Tmin = float(self.pg_tmin.get())
                Tmax = float(self.pg_tmax.get())
                Pmin = 0.0 
                Pmax = 20.0
                boundary_T, boundary_P = engine.run_phase_diagram(element, phase1, phase2, Tmin, Tmax, Pmin, Pmax)
                result["boundary_T"] = boundary_T
                result["boundary_P"] = boundary_P
                result["phase1"] = phase1
                result["phase2"] = phase2
                result["Tmin"] = Tmin
                result["Tmax"] = Tmax
                
            elif mode == "Vapor Pressure":
                tmin = float(self.vp_tmin.get())
                tmax = float(self.vp_tmax.get())
                T_arr, P_arr, H_vap = engine.run_vapor_pressure(atoms, tmin, tmax)
                result["T"] = T_arr
                result["P"] = P_arr
                result["H_vap"] = H_vap
                
            elif mode == "Thermodynamics":
                is_crystal = (phase != "molecule")
                T, G, S, Cp = engine.run_thermodynamics(atoms, is_crystal)
                result["T"] = T
                result["G"] = G
                result["S"] = S
                result["Cp"] = Cp

            # Send result to main thread
            self.task_queue.put({"status": "success", "data": result})
            
        except Exception as e:
            import traceback
            import sys
            print("\n" + "!"*60)
            print("BÁO CÁO TRUY VẾT LỖI (EXTREME DEBUG):")
            # 1. In toàn bộ cây phả hệ lỗi ra Terminal
            traceback.print_exc(file=sys.stdout)
            
            # 2. In thêm các biến môi trường quan trọng
            print(f"Lỗi tại Mode: {mode}")
            try:
                print(f"Cấu trúc Atoms: {atoms.get_chemical_formula()}")
            except: pass
            print(f"Kiểu dữ liệu e: {type(e)}")
            print("!"*60 + "\n")
            
            self.task_queue.put({"status": "error", "message": str(e)})

    def _process_queue(self):
        try:
            while True:
                msg = self.task_queue.get_nowait()
                
                self.is_running = False
                self.run_btn.configure(state="normal", text="▶ Run Simulation")
                self.progress_bar.stop()
                
                if msg["status"] == "error":
                    messagebox.showerror("Simulation Error", msg["message"])
                else:
                    self._update_gui_with_results(msg["data"])
                    
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_queue)

    def _update_gui_with_results(self, data):
        mode = data["mode"]
        element = data["element"]
        phase = data["phase"]
        
        if mode == "Equilibrium Scan":
            self.equi_ax.cla()
            a_vals = data["a_vals"]
            energies = data["energies"]
            self.equi_ax.plot(a_vals, energies, "o-", label=f"{element} ({phase})")
            self.equi_ax.set_xlabel("Lattice constant a (Å)")
            self.equi_ax.set_ylabel("Total Energy (eV)")
            self.equi_ax.set_title(f"Energy vs. a — {element} ({phase})")
            self.equi_ax.legend()
            self.equi_canvas.draw()
            
            # Update result label
            # CẢI TIẾN: Sử dụng Parabolic Fit để tìm điểm cực tiểu chính xác hơn np.argmin
            try:
                # Chỉ lấy các điểm xung quanh vùng có năng lượng thấp nhất để fit chính xác hơn
                idx_min = np.argmin(energies)
                # Fit bậc 2: E = Aa^2 + Ba + C
                coeffs = np.polyfit(a_vals, energies, 2)
                a_fit = -coeffs[1] / (2 * coeffs[0])
                E_fit = np.polyval(coeffs, a_fit)
                
                # Kiểm tra nếu điểm fit nằm ngoài dải quét thì fallback về argmin
                if a_fit < min(a_vals) or a_fit > max(a_vals):
                    a_opt, E_min = a_vals[idx_min], energies[idx_min]
                else:
                    a_opt, E_min = a_fit, E_fit
            except Exception:
                idx = np.argmin(energies)
                a_opt, E_min = a_vals[idx], energies[idx]

            self.equi_res_label.configure(text=f"Equilibrium Scan: Optimal a: {a_opt:.3f} Å  |  Min Energy: {E_min:.4f} eV")
            
        elif mode == "Equation of State":
            self.eos_ax.cla()
            self.eos_ax.plot(data["volumes"], data["energies"], 'o', label='Data')
            eos = data["eos"]
            try:
                v_fit = np.linspace(min(data["volumes"]), max(data["volumes"]), 100)
                e_fit = eos.eos(v_fit, eos.eos_string, eos.v0, eos.e0, eos.B, eos.B1)
                self.eos_ax.plot(v_fit, e_fit, '-', label='Fit')
            except Exception:
                pass
            self.eos_ax.set_xlabel("Volume (Å³)")
            self.eos_ax.set_ylabel("Total Energy (eV)")
            self.eos_ax.set_title(f"Equation of State — {element} ({phase})")
            self.eos_ax.legend()
            self.eos_canvas.draw()
            
            # Update result label
            self.eos_res_label.configure(text=f"Equation of State: Bulk Modulus (B0): {data['B_GPa']:.2f} GPa  |  Optimal V0: {data['v0']:.2f} Å³")
            messagebox.showinfo("Equation of State", f"Bulk Modulus: {data['B_GPa']:.2f} GPa\nOptimal Volume: {data['v0']:.4f} Å³")

        elif mode == "Tensile Test":
            self.tensile_ax.cla()
            strains = data["strains"]
            stresses = data["stresses"]
            self.tensile_ax.plot(strains * 100, stresses, 'o-r')
            self.tensile_ax.set_xlabel("Strain (%)")
            self.tensile_ax.set_ylabel("Stress (GPa)")
            self.tensile_ax.set_title(f"Tensile Test — {element} ({phase})")
            self.tensile_ax.grid(True)
            self.tensile_canvas.draw()
            
            # Update result label
            max_stress = max(stresses)
            max_strain = max(strains)
            avg_nu = data.get("avg_nu", 0.0)
            self.tensile_res_label.configure(text=f"Tensile Test: Max Stress (UTS): {max_stress:.2f} GPa  |  Max Strain: {max_strain:.3f}  |  Poisson Ratio: {avg_nu:.3f}")
            
            msg = "Tensile Test Completed"
            if len(strains) > 2:
                slope, _ = np.polyfit(strains[:3], stresses[:3], 1)
                msg += f"\nEstimated Young's Modulus: {slope:.2f} GPa"
                msg += f"\nAverage Poisson's Ratio (ν): {avg_nu:.3f}"
            messagebox.showinfo("Tensile Test", msg)

        elif mode == "Relaxation":
            self.relax_log.delete("1.0", tk.END)
            res = data["relax_results"]
            self.relax_log.insert(tk.END, f"Relaxation: Final Energy: {res['energy']:.4f} eV  |  Max Force: {res['fmax']:.4f} eV/Å\n")
            self.relax_log.insert(tk.END, f"Relaxation completed in {res['steps']} steps\n")
            
        elif mode == "Phonon":
            for widget in self.phonon_frame.winfo_children():
                widget.destroy()
            if not data["has_imag"]:
                messagebox.showinfo("Phonon", "No imaginary frequencies found.")
            canvas = FigureCanvasTkAgg(data["fig"], master=self.phonon_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()
            messagebox.showinfo("Phonon", f"Band structure plot saved as:\n{data['plot_path']}")
            
        elif mode == "Molecular Dynamics":
            self.md_fig.clear()
            # Create 3 subplots
            ax_e = self.md_fig.add_subplot(3, 1, 1)
            ax_t = self.md_fig.add_subplot(3, 1, 2)
            ax_v = self.md_fig.add_subplot(3, 1, 3)
            
            times = data["times"]
            
            ax_e.plot(times, data["energies"], 'b-')
            ax_e.set_ylabel('Energy (eV)')
            ax_e.set_title(f"{element} {phase} MD ({data['ensemble']} @ {data['pressure']} GPa)")
            ax_e.grid(True, alpha=0.3)
            
            ax_t.plot(times, data["temps"], 'r-')
            ax_t.set_ylabel('Temp (K)')
            ax_t.grid(True, alpha=0.3)
            
            ax_v.plot(times, data["volumes"], 'g-')
            ax_v.set_ylabel('Volume (Å³)')
            ax_v.set_xlabel('Time (fs)')
            ax_v.grid(True, alpha=0.3)
            
            self.md_fig.tight_layout()
            self.md_canvas.draw()
            
            final_v = data["volumes"][-1]
            final_t = data["temps"][-1]
            self.md_res_label.configure(text=f"Molecular Dynamics: Final Temp: {final_t:.1f} K  |  Final Volume: {final_v:.1f} Å³")
            
        elif mode == "Diffusion":
            self.diff_ax.cla()
            self.diff_ax.plot(data["times"], data["msd_vals"], 'b-', linewidth=2)
            self.diff_ax.set_xlabel("Time (ps)")
            self.diff_ax.set_ylabel("Mean Square Displacement (Å²)")
            self.diff_ax.set_title(f"Diffusion at {self.diff_temp.get()}K — {element} ({phase})")
            self.diff_ax.grid(True)
            self.diff_canvas.draw()
            self.diff_res_label.configure(text=f"Diffusion: Diffusion Coefficient (D): {data['D']:.4e} cm²/s")
            messagebox.showinfo("Diffusion", f"Estimated Diffusion Coefficient (D): {data['D']:.4e} cm²/s")

        elif mode == "Phase Diagram":
            self.phase_ax.cla()
            b_T = data["boundary_T"]
            b_P = data["boundary_P"]
            p1 = data["phase1"]
            p2 = data["phase2"]
            
            if len(b_T) > 0:
                # Sắp xếp các điểm theo thứ tự Áp suất tăng dần để nối nét liền
                sort_idx = np.argsort(b_P)
                sorted_T = np.array(b_T)[sort_idx]
                sorted_P = np.array(b_P)[sort_idx]

                # Thay vì plot(b_T, b_P), ta plot mảng đã sắp xếp
                self.phase_ax.plot(sorted_T, sorted_P, 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Phase Boundary')
                
                # Fill between colors using UI limits
                t_max_fill = data["Tmax"]
                t_min_fill = data["Tmin"]
                self.phase_ax.fill_betweenx(sorted_P, sorted_T, t_max_fill, color='#3498db', alpha=0.2, label=f'Region {p2}')
                self.phase_ax.fill_betweenx(sorted_P, t_min_fill, sorted_T, color='#2ecc71', alpha=0.2, label=f'Region {p1}')

            # THÊM DÒNG NÀY ĐỂ ÉP ĐỒ THỊ KHÓA TRỤC X THEO Ô NHẬP LIỆU:
            self.phase_ax.set_xlim([data["Tmin"], data["Tmax"]])

            self.phase_ax.set_xlabel("Temperature (K)")
            self.phase_ax.set_ylabel("Pressure (GPa)")
            self.phase_ax.set_title(f"Phase Diagram: {p1} vs {p2} ({element})")
            self.phase_ax.legend()
            self.phase_canvas.draw()
            
        elif mode == "Vapor Pressure":
            self.vp_ax.cla()
            self.vp_ax.plot(data["T"], data["P"], 'm-', linewidth=2)
            self.vp_ax.set_xlabel("Temperature (K)")
            self.vp_ax.set_ylabel("Vapor Pressure (Pa)")
            self.vp_res_label.configure(text=f"Vapor Pressure: Cohesive Energy: {data['H_vap']:.3f} eV/atom")
            self.vp_ax.set_title(f"Vapor Pressure P(T) — {element}")
            self.vp_ax.set_yscale('log') # Thường vẽ thang log cho áp suất hơi
            self.vp_ax.grid(True, which="both", ls="-", alpha=0.5)
            self.vp_res_label.configure(text=f"Estimated Cohesive Energy (Delta H_vap): {data['H_vap']:.3f} eV/atom")
            self.vp_canvas.draw()
            
        elif mode == "Thermodynamics":
            self.thermo_ax.cla()
            # Remove any old twin axis
            for ax in self.thermo_fig.get_axes():
                if ax is not self.thermo_ax:
                    ax.remove()
            
            ax1 = self.thermo_ax
            ax2 = ax1.twinx()
            
            # BRUTAL CASTING AND FLATTENING
            T = np.array(data["T"], dtype=float).flatten()
            G = np.array(data["G"], dtype=float).flatten()
            S = np.array(data["S"], dtype=float).flatten()
            Cv = np.array(data["Cp"], dtype=float).flatten()
            
            ln1 = ax1.plot(T, G, 'b-', linewidth=2, label="Helmholtz F (eV)")
            ln2 = ax2.plot(T, S, color='orange', linewidth=1.5, label="Entropy S (eV/K)")
            ln3 = ax2.plot(T, Cv, 'g-', linewidth=1.5, label="Heat Capacity Cv (eV/K)")
            
            ax1.set_xlabel("Temperature (K)")
            ax1.set_ylabel("Helmholtz Free Energy (eV)", color='b')
            ax2.set_ylabel("Entropy / Cv (eV/K)", color='darkorange')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            
            # Combined legend
            lns = ln1 + ln2 + ln3
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left', fontsize=8)
            ax1.set_title(f"Thermodynamic Properties: {data['formula']}")
            self.thermo_canvas.draw()

    def _safe_exit(self):
        if self.is_running:
            if messagebox.askokcancel("Quit", "Calculations running! Force quit?"):
                self.destroy()
        else:
            self.destroy()

if __name__ == "__main__":
    app = MatterSimApp()
    app.mainloop()
