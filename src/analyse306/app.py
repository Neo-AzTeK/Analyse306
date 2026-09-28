import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import MATER_306
from pathlib import Path
from typing import Tuple, List, Dict, Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import json

# ----------------------
# Config & Persistence
# ----------------------
CONFIG_FILE = "last_session.json"
OLD_CONFIG_FILE = "params_mater306.json"

def save_config(params: Dict[str, float], file_paths: Dict[str, str]):
    try:
        data = {
            "parameters": params,
            "last_files": file_paths
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[WARNING] Impossible de sauvegarder la configuration : {e}")

def load_config() -> Tuple[Dict[str, float], Dict[str, str]]:
    global PARAMETERS_MAT306
    initial_paths = {"tension": "", "shear": "", "mixed": ""}
    
    # Migration depuis l'ancien nom de fichier s'il existe
    if not os.path.exists(CONFIG_FILE) and os.path.exists(OLD_CONFIG_FILE):
        try:
            os.rename(OLD_CONFIG_FILE, CONFIG_FILE)
            print(f"[INFO] Migration de {OLD_CONFIG_FILE} vers {CONFIG_FILE}")
        except Exception as e:
            print(f"[WARNING] Échec de la migration du fichier config : {e}")

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                # Gestion de la transition depuis l'ancien format plat
                if isinstance(data, dict) and "parameters" in data:
                    PARAMETERS_MAT306.update(data["parameters"])
                    initial_paths.update(data.get("last_files", {}))
                else:
                    PARAMETERS_MAT306.update(data)
                print(f"[INFO] Configuration chargée depuis {CONFIG_FILE}")
        except Exception as e:
            print(f"[WARNING] Impossible de charger la configuration : {e}")
    return PARAMETERS_MAT306, initial_paths

# ----------------------
# Variables
# ----------------------

COL_DISP_IDX = 1
COL_FORCE_IDX = 2

essais: Dict[str, Dict[str, Any]] = {
    "tension": {
        "nom_essai": "Tension",
        "filename": "",
        "angle": 0,
        "nom_mat_cols": ["f_n", "d_n"]
    },
    "shear": {
        "nom_essai": "Cisaillement",
        "filename": "",
        "angle": 90,
        "nom_mat_cols": ["f_t", "d_t"]
    },
    "mixed": {
        "nom_essai": "Mixte",
        "filename": "",
        "angle": 45,
        "nom_mat_cols": ["force", "delta"]
    }
}

PARAMETERS_MAT306 = {
    'f_n_max':5450.0, 'xi_n':0.95, 'd_n_fail':18,
    'f_t_max':9000.0, 'xi_t':0.95, 'd_t_fail':7.5,
    'alpha_1':0.5, 'alpha_2':0.75, 'alpha_3':0.85,
    'h_ms':4.0
}

def recup_unites(filename: Path) -> int:
    # -------------------------------------------------
    # 1) Lecture de la ligne des unités
    # -------------------------------------------------
    units = pd.read_csv(
        filename,
        sep=";",
        encoding="windows-1252",
        decimal=",",
        nrows=1,
        skiprows=1
    )

    unite_force = str(units.columns[COL_FORCE_IDX]).strip()  # colonne Force

    # -------------------------------------------------
    # 2) Détermination du facteur de conversion → N
    # -------------------------------------------------
    if "kN" in unite_force:
        facteur = 1000
    elif "daN" in unite_force:
        facteur = 10
    elif "N" in unite_force:
        facteur = 1
    else:
        raise ValueError(f"Unité de force inconnue : {unite_force}")

    # ✅ Affichage de l’unité détectée
    print(
        f"[INFO] Fichier : {filename.name} | "
        f"Unité force détectée : {unite_force} -> conversion x{facteur} (N)"
    )
    return facteur

def load_essai_data(filename: Path) -> pd.DataFrame:
    """Load the CSV file skipping the units row (index 1)."""
    return pd.read_csv(
        filename,
        sep=";",
        encoding="windows-1252",
        decimal=",",
        skiprows=[1]
    )

def ajoute_essai(df: pd.DataFrame, type_essai: str, facteur: int, axe_essais: plt.Axes) -> None:
    x = df.iloc[:, COL_DISP_IDX]
    y = df.iloc[:, COL_FORCE_IDX] * facteur

    axe_essais.plot(x, y, label=type_essai)

def trace_essais_comp(df: pd.DataFrame, nom_essai: str, facteur: int, ax: plt.Axes, simulation: pd.DataFrame, nom_cols: List[str], label_essai: str = None, label_simu: str = None) -> Tuple[float, float, float, float]:
    # ESSAI
    x_essai = df.iloc[:, COL_DISP_IDX]                 # déplacement (mm)
    y_essai = df.iloc[:, COL_FORCE_IDX] * facteur       # effort (N)

    l_essai = label_essai if label_essai else f"Essai: {nom_essai}"
    ax.plot(x_essai, y_essai, label=l_essai, color="black", alpha=0.5)

    # SIMU
    x_simu = simulation[nom_cols[1]]       # déplacement (mm)
    y_simu = simulation[nom_cols[0]]       # effort (N)

    l_simu = label_simu if label_simu else f"MATER306: {nom_essai}"
    ax.plot(x_simu, y_simu, label=l_simu, color="blue", linestyle='dashed')

    ax.set_title(nom_essai)
    ax.set_xlabel('Déplacement (mm)')
    ax.set_ylabel('Force (N)')
    ax.grid(True)
    return max(y_essai), max(x_essai), max(y_simu), max(x_simu)

def trace_global(axes_corr: List[plt.Axes], labels: List[str], x_efforts: List[float], y_efforts: List[float], x_disp: List[float], y_disp: List[float]) -> None:
    # ---------------------------------------------------------
    # 7B) Corrélation efforts max
    # ---------------------------------------------------------
    ax = axes_corr[1]
    ax.scatter(x_efforts, y_efforts, color="blue")

    for i, label in enumerate(labels):
        ax.annotate(label, (x_efforts[i], y_efforts[i]), xytext=(5,5), textcoords="offset points")

    if x_efforts and y_efforts:
        max_val = np.max([np.max(x_efforts), np.max(y_efforts)])
        ax.plot([0, max_val], [0, max_val], "--", color="black")
        
        # Régression
        if len(x_efforts) > 1:
            a, b = np.polyfit(x_efforts, y_efforts, 1)
            y_pred = a * np.array(x_efforts) + b
            R2_effort = 1 - np.sum((np.array(y_efforts) - y_pred)**2) / np.sum((np.array(y_efforts) - np.mean(y_efforts))**2)
            ax.text(0.05*max_val, 0.9*max_val, f"$R^2 = {R2_effort:.3f}$", color="red")

    ax.set_title("Efforts max : modèle vs essais")
    ax.set_xlabel("Essais (N)")
    ax.set_ylabel("Modèle (N)")
    ax.grid(True)

    # ---------------------------------------------------------
    # 7C) Corrélation déplacements max
    # ---------------------------------------------------------
    ax = axes_corr[2]
    ax.scatter(x_disp, y_disp, color="green")

    for i, label in enumerate(labels):
        ax.annotate(label, (x_disp[i], y_disp[i]), xytext=(5,5), textcoords="offset points")

    if x_disp and y_disp:
        max_val = np.max([np.max(x_disp), np.max(y_disp)])
        ax.plot([0, max_val], [0, max_val], "--", color="black")
        
        # Régression
        if len(x_disp) > 1:
            a, b = np.polyfit(x_disp, y_disp, 1)
            y_pred = a * np.array(x_disp) + b
            R2_disp = 1 - np.sum((np.array(y_disp) - y_pred)**2) / np.sum((np.array(y_disp) - np.mean(y_disp))**2)
            ax.text(0.05*max_val, 0.9*max_val, f"$R^2 = {R2_disp:.3f}$", color="red")

    ax.set_title("Déplacements max : modèle vs essais")
    ax.set_xlabel("Essais (mm)")
    ax.set_ylabel("Modèle (mm)")
    ax.grid(True)

def faire_tableau_recap(labels: List[str], x_efforts: List[float], y_efforts: List[float], x_disp: List[float], y_disp: List[float], save: bool = True) -> plt.Figure:
    # ---------------------------------------------------------
    # 9) Tableaux récapitulatifs du bilan des ecarts essais/modèles
    # ---------------------------------------------------------

    # --- TABLEAU EFFORTS ---
    df_efforts = pd.DataFrame({
        "Type d'essai": labels,
        "Essai (N)": x_efforts,
        "Modèle (N)": y_efforts,
    })

    # Écarts
    df_efforts["Écart abs (N)"] = abs(df_efforts["Modèle (N)"] - df_efforts["Essai (N)"])
    if not df_efforts["Essai (N)"].eq(0).all():
        df_efforts["Écart (%)"] = 100 * df_efforts["Écart abs (N)"] / df_efforts["Essai (N)"]

    # --- TABLEAU DÉPLACEMENTS ---
    df_depl = pd.DataFrame({
        "Type d'essai": labels,
        "Essai (mm)": x_disp,
        "Modèle (mm)": y_disp,
    })

    # Écarts
    df_depl["Écart abs (mm)"] = abs(df_depl["Modèle (mm)"] - df_depl["Essai (mm)"])
    if not df_depl["Essai (mm)"].eq(0).all():
        df_depl["Écart (%)"] = 100 * df_depl["Écart abs (mm)"] / df_depl["Essai (mm)"]

    # --- ARRONDIS ---
    df_efforts = df_efforts.round(2)
    df_depl = df_depl.round(2)

    # --- AFFICHAGE DES 2 TABLEAUX CÔTE À CÔTÉ ---
    fig, axs = plt.subplots(1, 2, num=8, clear=True, figsize=(16, 2))
    titles = ["Bilan sur les efforts max", "Bilan sur les déplacements max"]
    dataframes = [df_efforts, df_depl]

    for ax, title, df in zip(axs, titles, dataframes):
        ax.axis("off")
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc="center",
            cellLoc="center" # centers text
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 2.2) # better height padding

        # Styling cells iteratively
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#E0E0E0')
            if r == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2C3E50') # Elegant dark blue/grey header
            else:
                # Alternating faint row colors
                cell.set_facecolor('#F8F9FA' if r % 2 == 0 else '#FFFFFF')

                # Highlight the "Écart (%)" gracefully
                if c == len(df.columns) - 1: # Assuming exact last column is Écart %
                    try:
                        val = float(df.iloc[r-1, c])
                        if val > 15:
                            cell.set_text_props(color='#E74C3C', weight='bold') # soft red
                        elif val < 5:
                            cell.set_text_props(color='#27AE60', weight='bold') # soft green
                    except:
                        pass
        
        ax.set_title(title, pad=10, fontsize=14, weight='bold', color='#34495E')
        table.auto_set_column_width(col=list(range(len(df.columns))))

    fig.tight_layout()
    
    if save:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "bilan_essais_simu.png", dpi=300, bbox_inches='tight')
    
    return fig

def plot_polar_failure_criteria(tension_force: float, shear_force: float, mixed_force: float, save: bool = True) -> plt.Figure:
    # Ordonner les valeurs par angle : 0° (Tension), 45° (Mixed), 90° (Shear)
    angles_deg = np.array([0, 45, 90])
    angles_rad = np.deg2rad(angles_deg)
    efforts = np.array([round(tension_force), round(mixed_force), round(shear_force)])

    theta = np.linspace(0, np.pi/2, 300)
    efforts_interp = np.interp(theta, angles_rad, efforts)

    sigma1 = tension_force
    sigma2 = shear_force
    rayon_mohr = (sigma1 - sigma2) / 2
    centre_mohr = (sigma1 + sigma2) / 2
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # --- Graph 1 : Cercle de Mohr classique sur fond coloré ---
    fig1 = plt.figure(num=6, clear=True, figsize=(7, 6))
    ax1 = fig1.add_subplot(111, projection='polar')
    ax1.set_thetamin(0)
    ax1.set_thetamax(90)
    
    angles_ticks = [0, 15, 30, 45, 60, 75, 90]
    ax1.set_thetagrids(angles_ticks, labels=["Tension (0°)", "15°", "30°", "45°", "60°", "75°", "Cisaillement (90°)"])

    ax1.fill_between(theta, 0, efforts_interp, color="lightgreen", alpha=0.6)
    ax1.fill_between(theta, efforts_interp, efforts_interp.max()*1.2, color="orange", alpha=0.4)
    ax1.plot(theta, efforts_interp, color="blue", linewidth=2, label="Critère de rupture expérimental")
    
    ax1.scatter(angles_rad, efforts, color='red', zorder=5)
    for ang, eff in zip(angles_deg, efforts):
        ax1.text(np.deg2rad(ang), eff + max(efforts)*0.08, f"{eff} N", ha='center')

    sigma_polaire = centre_mohr + rayon_mohr * np.cos(2 * theta)
    ax1.plot(theta, sigma_polaire, color="blue", linestyle="--", label="Critère de Mohr")
    
    ax1.set_title("Critère de rupture basé sur les essais (0-45-90)°\n", weight='bold')
    ax1.legend(loc="upper right")

    fig1.tight_layout()
    if save:
        fig1.savefig(output_dir / "mohr_colore.png", dpi=300, bbox_inches='tight')
    return fig1


def read_experiments(file_tens: str, file_shear: str, file_mixed: str):
    """Read all three tests without evaluating any model parameters."""
    data = {}
    for name, filename in zip(essais, (file_tens, file_shear, file_mixed)):
        path = Path(filename)
        if not filename or not path.is_file():
            raise ValueError(f"Fichier d'essai introuvable : {filename}")
        factor = recup_unites(path)
        frame = load_essai_data(path)
        values = frame.iloc[:, [COL_DISP_IDX, COL_FORCE_IDX]].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Données d'essai vides ou invalides : {filename}")
        data[name] = (frame, factor)
    return data


def experimental_analysis(file_tens: str, file_shear: str, file_mixed: str,
                          save_outputs: bool = True, embedded: bool = False):
    """Create only the raw curves, peak table and experimental polar plot."""
    plt.close('all')
    try:
        data = read_experiments(file_tens, file_shear, file_mixed)
        fig_raw, ax = plt.subplots(num=1, clear=True)
        fig_max, ax_table = plt.subplots(num=2, clear=True, figsize=(6, 3))
        peaks = {}
        cells = []
        for name, (frame, factor) in data.items():
            meta = essais[name]
            ajoute_essai(frame, meta['nom_essai'], factor, ax)
            peaks[name] = max(frame.iloc[:, COL_FORCE_IDX] * factor)
            cells.append([meta['nom_essai'], round(peaks[name])])
        ax.set(title="Courbes d'essai de traction", xlabel="Déplacement (mm)", ylabel="Effort (N)")
        ax.legend()
        ax.grid(True)
        ax_table.axis('off')
        ax_table.set_title("Tenue mécanique en N")
        table = ax_table.table(cellText=cells, colLabels=["Essai", "Tenue mécanique (N)"],
                               loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.2)
        fig_raw.tight_layout()
        fig_max.tight_layout()
        fig_polar = plot_polar_failure_criteria(peaks['tension'], peaks['shear'], peaks['mixed'], save=save_outputs)
        if save_outputs:
            output = Path('output')
            output.mkdir(exist_ok=True)
            fig_raw.savefig(output / 'courbes_essais.png', dpi=300, bbox_inches='tight')
            fig_max.savefig(output / 'tableau_rivet_bleu.png', dpi=300, bbox_inches='tight')
        figures = [fig_raw, fig_max, fig_polar]
        if embedded:
            for fig in figures:
                plt.close(fig)
        else:
            plt.show(block=False)
        return figures
    except Exception:
        plt.close('all')
        raise


def main(file_tens: str, file_shear: str, file_mixed: str, save_outputs: bool = True, preview_mode: bool = False):
    # Fermer les anciennes fenêtres avant de commencer
    plt.close('all')
    
    data = read_experiments(file_tens, file_shear, file_mixed)

    fig_width = 10 if preview_mode else 18
    fig_height = 3 if preview_mode else 5
    fig_comp, (ax_tens, ax_shear, ax_mixed) = plt.subplots(1, 3, num=4, clear=True, figsize=(fig_width, fig_height))
    
    if not preview_mode:
        fig_corr, axes_corr = plt.subplots(1, 3, num=5, clear=True, figsize=(18, 5)) # Subplots corrélation efforts + déplacements

    comparison_axes = dict(zip(essais, (ax_tens, ax_shear, ax_mixed)))

    labels_list: List[str] = []
    x_efforts_list: List[float] = []
    y_efforts_list: List[float] = []
    x_disp_list: List[float] = []
    y_disp_list: List[float] = []

    for name, essai_meta in essais.items():
        df_essai, facteur = data[name]
        try:
            simulation = MATER_306.compute_response(90 - essai_meta["angle"], PARAMETERS_MAT306)

            # Tracé des essais et du MAT306 dans le fig_comp
            f_max_essai, d_max_essai, f_max_simu, d_max_simu = trace_essais_comp(
                df_essai,
                essai_meta["nom_essai"],
                facteur,
                comparison_axes[name],
                simulation,
                essai_meta["nom_mat_cols"]
            )
            
            if not preview_mode:
                # Tracé comparatif superposition dans fig_corr[0]
                trace_essais_comp(
                    df_essai,
                    "Essais VS MATER306",
                    facteur,
                    axes_corr[0],
                    simulation,
                    essai_meta["nom_mat_cols"],
                    label_essai="Essais",
                    label_simu="MATER306"
                )
        
                # Store maxes for correlation charts
                labels_list.append(essai_meta["nom_essai"])
                x_efforts_list.append(f_max_essai)
                y_efforts_list.append(f_max_simu)
                x_disp_list.append(d_max_essai)
                y_disp_list.append(d_max_simu)

        except Exception as e:
            plt.close("all")
            raise RuntimeError(f"Simulation impossible pour {essai_meta['nom_essai']} : {e}") from e

    # S'il y a des données valides, on plot
    if not labels_list and not preview_mode:
        print("[WARNING] Aucune donnée valide trouvée pour générer les graphiques finaux.")
        return

    if not preview_mode:
        # Plot correlations and final table
        trace_global(axes_corr, labels_list, x_efforts_list, y_efforts_list, x_disp_list, y_disp_list)
        faire_tableau_recap(labels_list, x_efforts_list, y_efforts_list, x_disp_list, y_disp_list, save=save_outputs)
    
    fig_comp.tight_layout()
    for ax in (ax_tens, ax_shear, ax_mixed):
        ax.legend()

    if not preview_mode:
        axes_corr[0].set_title("Essais VS MATER306")
        
        # Éviter les légendes dupliquées sur la superposition
        handles, legends = axes_corr[0].get_legend_handles_labels()
        by_label = dict(zip(legends, handles))
        axes_corr[0].legend(by_label.values(), by_label.keys(), loc='upper right')
    
        fig_corr.tight_layout()
        
        if save_outputs:
            fig_corr.savefig(Path("output") / "correlation_efforts.png", dpi=300, bbox_inches='tight')
            fig_comp.savefig(Path("output") / "comparaison_matis.png", dpi=300, bbox_inches='tight')
        
        # --- Nouveau : Tableau des paramètres de simulation ---
        # On retire h_ms pour le tableau horizontal du rapport
        table_params = {k: v for k, v in PARAMETERS_MAT306.items() if k != 'h_ms'}
        p_headers = list(table_params.keys())
        p_values = [[v for v in table_params.values()]]
        
        # Adaptation de figsize au nombre de colonnes
        width = len(p_headers)
        fig_params, ax_p = plt.subplots(num=3, clear=True, figsize=(width, 2))
        ax_p.axis('off')
        
        p_table = ax_p.table(
            cellText=p_values,
            colLabels=p_headers,
            loc="center",
            cellLoc="center"
        )
        p_table.auto_set_font_size(False)
        p_table.set_fontsize(10)
        p_table.scale(1.0, 2.5) # Plus de hauteur pour les cellules
        p_table.auto_set_column_width(col=list(range(len(p_headers)))) # Adaptation automatique de la largeur
        
        # Styling du tableau
        for (r, c), cell in p_table.get_celld().items():
            cell.set_edgecolor('#E0E0E0')
            if r == 0: # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2C3E50')
            else: # Values
                cell.set_facecolor('#FFFFFF')
                
        ax_p.set_title("Paramètres de Simulation MAT306", weight='bold', pad=20, fontsize=14, color='#34495E')
        fig_params.tight_layout()
        if save_outputs:
            fig_params.savefig(Path("output") / "parametres_simulation.png", dpi=300, bbox_inches='tight')

    if not preview_mode:
        plt.show(block=False)
    else:
        plt.close(fig_comp)
        return fig_comp

def launch_gui(initial_paths: Dict[str, str]):
    root = tk.Tk()
    root.title("Rapport d'essais - Sélection des essais (Tension, Cisaillement, Mixte)")
    
    # Taille adaptative
    root.geometry("800x650") # Taille par défaut suggestive
    root.resizable(True, True)
    
    # Forcer l'adaptation au contenu
    root.update_idletasks()
    root.geometry("") 
    root.minsize(800, 400)

    paths = initial_paths.copy()
    
    # Dictionnaire pour stocker les widgets Entry des paramètres
    entry_widgets = {}
    
    # Frame pour l'aperçu des graphes (non affiché par défaut)
    preview_frame = tk.LabelFrame(root, text=" 3. Aperçu des Graphes ", padx=10, pady=10, fg="#3A6996", font=("Helvetica", 10, "bold"))

    def select_file(test_type, label):
        filename = filedialog.askopenfilename(
            title=f"Sélectionnez le fichier pour l'essai: {test_type}",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            paths[test_type] = filename
            label.config(text=Path(filename).name, fg="#2C3E50")

    embedded_canvases = []

    def clear_display():
        for canvas in embedded_canvases:
            canvas.get_tk_widget().destroy()
            plt.close(canvas.figure)
        embedded_canvases.clear()
        for widget in preview_frame.winfo_children():
            widget.destroy()
        preview_frame.pack_forget()
        plt.close('all')

    def embed_figures(figures, titles):
        notebook = ttk.Notebook(preview_frame)
        notebook.pack(fill='both', expand=True)
        for fig, title in zip(figures, titles):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=title)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            embedded_canvases.append(canvas)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        preview_frame.pack(fill='both', expand=True, padx=20, pady=10)

    def validate_paths():
        if not all(paths.get(name) for name in essais):
            messagebox.showerror("Erreur de sélection", "Veuillez sélectionner les 3 fichiers d'essai.")
            return False
        return True

    def run_experiments(embedded=False):
        if not validate_paths():
            return
        clear_display()
        try:
            save_config(PARAMETERS_MAT306, paths)
            figures = experimental_analysis(paths['tension'], paths['shear'], paths['mixed'],
                                            save_outputs=not embedded, embedded=embedded)
            if embedded:
                embed_figures(figures, ["Courbes d'essai", "Tenue mécanique", "Critère de rupture"])
        except Exception as exc:
            clear_display()
            messagebox.showerror("Erreur des essais", str(exc))

    def run_analysis(save_files=True):
        if not validate_paths():
            return
        try:
            params = {key: float(entry_widgets[key].get().replace(',', '.')) for key in PARAMETERS_MAT306}
            if not all(np.isfinite(value) for value in params.values()):
                raise ValueError("Les paramètres doivent être des nombres finis.")
        except ValueError as exc:
            messagebox.showerror("Erreur de saisie", f"Paramètre MAT306 invalide : {exc}")
            return
        PARAMETERS_MAT306.update(params)
        if save_files:
            save_config(PARAMETERS_MAT306, paths)
        clear_display()
        try:
            fig = main(paths['tension'], paths['shear'], paths['mixed'],
                       save_outputs=save_files, preview_mode=not save_files)
            if not save_files:
                embed_figures([fig], ["Essais vs Modèle"])
        except Exception as exc:
            clear_display()
            messagebox.showerror("Erreur de simulation", str(exc))

    def close_window():
        clear_display()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', close_window)

    # --- Section Sélection des Fichiers ---
    files_frame = tk.LabelFrame(root, text=" 1. Sélection des Fichiers d'Essai ", padx=10, pady=10, fg="#3A6996", font=("Helvetica", 10, "bold"))
    files_frame.pack(fill="x", padx=20, pady=10)

    # 1. Tension config
    tk.Label(files_frame, text="Tension (90°):", width=15, anchor='w', font=("Helvetica", 9, "bold")).grid(row=0, column=0, pady=5)
    
    txt_tens = Path(paths["tension"]).name if paths["tension"] else "Aucun fichier..."
    fg_tens = "#2C3E50" if paths["tension"] else "black"
    lbl_tens = tk.Label(files_frame, text=txt_tens, width=45, bg="#ecf0f1", anchor='w', fg=fg_tens)
    lbl_tens.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(files_frame, text="Parcourir", command=lambda: select_file("tension", lbl_tens), bg="#bdc3c7").grid(row=0, column=2)

    # 2. Shear config
    tk.Label(files_frame, text="Cisaillement (0°):", width=15, anchor='w', font=("Helvetica", 9, "bold")).grid(row=1, column=0, pady=5)
    
    txt_shear = Path(paths["shear"]).name if paths["shear"] else "Aucun fichier..."
    fg_shear = "#2C3E50" if paths["shear"] else "black"
    lbl_shear = tk.Label(files_frame, text=txt_shear, width=45, bg="#ecf0f1", anchor='w', fg=fg_shear)
    lbl_shear.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(files_frame, text="Parcourir", command=lambda: select_file("shear", lbl_shear), bg="#bdc3c7").grid(row=1, column=2)

    # 3. Mixed config
    tk.Label(files_frame, text="Mixte (45°):", width=15, anchor='w', font=("Helvetica", 9, "bold")).grid(row=2, column=0, pady=5)
    
    txt_mixed = Path(paths["mixed"]).name if paths["mixed"] else "Aucun fichier..."
    fg_mixed = "#2C3E50" if paths["mixed"] else "black"
    lbl_mixed = tk.Label(files_frame, text=txt_mixed, width=45, bg="#ecf0f1", anchor='w', fg=fg_mixed)
    lbl_mixed.grid(row=2, column=1, padx=10, pady=5)
    tk.Button(files_frame, text="Parcourir", command=lambda: select_file("mixed", lbl_mixed), bg="#bdc3c7").grid(row=2, column=2)

    files_actions = tk.Frame(files_frame)
    files_actions.grid(row=0, column=3, rowspan=3, sticky='e', padx=(20, 0))
    files_frame.columnconfigure(3, weight=1)
    tk.Button(files_actions, text="Aperçu des Essais", command=lambda: run_experiments(embedded=True),
              bg="#607D8B", fg="white", width=20).pack(pady=5)
    tk.Button(files_actions, text="Afficher les Essais", command=run_experiments,
              bg="#4A90E2", fg="white", width=20).pack(pady=5)

    # --- Section Paramètres MAT306 ---
    params_frame = tk.LabelFrame(root, text=" 2. Paramètres MAT306 ", padx=10, pady=10, fg="#3A6996", font=("Helvetica", 10, "bold"))
    params_frame.pack(fill="x", padx=20, pady=10)

    def add_param(parent, row, col, label, key):
        tk.Label(parent, text=label, anchor='w').grid(row=row, column=col, padx=5, pady=2, sticky='w')
        ent = tk.Entry(parent, width=8)
        ent.insert(0, str(PARAMETERS_MAT306[key]))
        ent.grid(row=row, column=col+1, padx=5, pady=2)
        entry_widgets[key] = ent

    # Groupes de paramètres
    # Tension
    tk.Label(params_frame, text="Tension:", font=("Helvetica", 9, "underline bold")).grid(row=0, column=0, sticky='w', pady=(0,5))
    add_param(params_frame, 1, 0, "f_n_max:", "f_n_max")
    add_param(params_frame, 2, 0, "xi_n:", "xi_n")
    add_param(params_frame, 3, 0, "d_n_fail:", "d_n_fail")

    # Shear
    tk.Label(params_frame, text="Cisaillement:", font=("Helvetica", 9, "underline bold")).grid(row=0, column=2, sticky='w', pady=(0,5), padx=(20,0))
    add_param(params_frame, 1, 2, "f_t_max:", "f_t_max")
    add_param(params_frame, 2, 2, "xi_t:", "xi_t")
    add_param(params_frame, 3, 2, "d_t_fail:", "d_t_fail")

    # Interaction & Global
    tk.Label(params_frame, text="Interaction / Global:", font=("Helvetica", 9, "underline bold")).grid(row=0, column=4, sticky='w', pady=(0,5), padx=(20,0))
    add_param(params_frame, 1, 4, "alpha_1:", "alpha_1")
    add_param(params_frame, 2, 4, "alpha_2:", "alpha_2")
    add_param(params_frame, 3, 4, "alpha_3:", "alpha_3")
    add_param(params_frame, 4, 4, "h_ms:", "h_ms")

    # Buttons
    btn_frame = tk.Frame(params_frame)
    btn_frame.grid(row=0, column=6, rowspan=5, sticky='e', padx=(20, 0))
    params_frame.columnconfigure(6, weight=1)

    btn_preview = tk.Button(btn_frame, text="Aperçu Rapide", font=("Helvetica", 12, "bold"), bg="#607D8B", fg="white", command=lambda: run_analysis(save_files=False), height=2, width=15)
    btn_preview.pack(pady=5, fill="x")

    btn_run = tk.Button(btn_frame, text="Lancer l'Analyse", font=("Helvetica", 12, "bold"), bg="#4A90E2", fg="white", command=lambda: run_analysis(save_files=True), height=2, width=20)
    btn_run.pack(pady=5, fill="x")

    root.mainloop()

if __name__ == "__main__":
    _, initial_paths = load_config()
    launch_gui(initial_paths)
