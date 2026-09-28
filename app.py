import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import MATER_306
from pathlib import Path
from typing import Tuple, List, Dict, Any
import tkinter as tk
from tkinter import filedialog, messagebox
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

def faire_tableau_recap(labels: List[str], x_efforts: List[float], y_efforts: List[float], x_disp: List[float], y_disp: List[float], save: bool = True) -> None:
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
    fig, axs = plt.subplots(1, 2, figsize=(16, 2))
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
    
    fig.show()

def plot_polar_failure_criteria(tension_force: float, shear_force: float, mixed_force: float, save: bool = True) -> None:
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
    fig1 = plt.figure(figsize=(7, 6))
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

    # --- Graph 2 : Graphique polaire simple ---
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111, projection='polar')
    ax2.set_thetamin(0)
    ax2.set_thetamax(90)
    
    angles_ticks = [0, 15, 30, 45, 60, 75, 90]
    ax2.set_thetagrids(angles_ticks, labels=["Tension (0°)", "15°", "30°", "45°", "60°", "75°", "Cisaillement (90°)"])
    
    ax2.plot(theta, sigma_polaire, color="blue", linestyle="--", label="Critère de Mohr")
    ax2.plot(theta, efforts_interp, label="Critère de rupture (exemple)")
    ax2.scatter(angles_rad, efforts, color='red', zorder=5)
    
    for ang, eff in zip(angles_deg, efforts):
        ax2.text(np.deg2rad(ang), eff - max(efforts)*0.08, f"{eff} N", ha='center')
        
    ax2.set_title("Critère de rupture basé sur les essais (0-45-90)°\n", weight='bold')
    ax2.grid(True)
    ax2.legend()
    
    fig1.tight_layout()
    fig2.tight_layout()
    
    if save:
        fig1.savefig(output_dir / "mohr_colore.png", dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / "mohr_simple.png", dpi=300, bbox_inches='tight')

def main(file_tens: str, file_shear: str, file_mixed: str, save_outputs: bool = True, preview_mode: bool = False):
    # Fermer les anciennes fenêtres avant de commencer
    plt.close('all')
    
    # Overwrite the global configs dynamically
    essais["tension"]["filename"] = file_tens
    essais["shear"]["filename"] = file_shear
    essais["mixed"]["filename"] = file_mixed

    if not preview_mode:
        fig_essais, axe_essais = plt.subplots() # Subplots basique des essais
        # --- Tableau des max ---
        fig_max, axe_tableau = plt.subplots(figsize=(5, 2))
        # Préférences
        header_color = "#4A90E2"
        cell_color = "#C0D7F5"
        cell_colors = [[cell_color, cell_color] for i in range(3)]
        # En-têtes
        headers = ["Essai", "Tenue mécanique\n(en N)"]
        cells = []
    
    fig_width = 10 if preview_mode else 18
    fig_height = 3 if preview_mode else 5
    fig_comp, (ax_tens, ax_shear, ax_mixed) = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    
    if not preview_mode:
        fig_corr, axes_corr = plt.subplots(1, 3, figsize=(18, 5)) # Subplots corrélation efforts + déplacements

    # Assign dynamically instantiated axes to the global configuration dictionary
    essais["tension"]["axe"] = ax_tens
    essais["shear"]["axe"] = ax_shear
    essais["mixed"]["axe"] = ax_mixed

    labels_list: List[str] = []
    x_efforts_list: List[float] = []
    y_efforts_list: List[float] = []
    x_disp_list: List[float] = []
    y_disp_list: List[float] = []

    for name, essai_meta in essais.items():
        file_path = Path(essai_meta["filename"])

        if not file_path.exists():
            print(f"[ERROR] Le fichier {file_path} est introuvable. Ignoré.")
            continue

        try:
            # 1. Calcul facteur depuis les untiés
            facteur = recup_unites(file_path)
    
            # 2. Extract Dataframe une seule fois pour tout l'itération
            df_essai = load_essai_data(file_path)
    
            # 3. Simulation MAT306 (Compensation inverse `90 - angle` requise par la config externe)
            simulation = MATER_306.compute_response(90 - essai_meta["angle"], PARAMETERS_MAT306)
    
            if not preview_mode:
                # Tracé des essais seuls
                ajoute_essai(df_essai, essai_meta["nom_essai"], facteur, axe_essais)
        
                # Faire tableau bleu
                max_effort = round(max(df_essai.iloc[:, COL_FORCE_IDX] * facteur))
                row = [f"{essai_meta['nom_essai']} ({essai_meta['angle']}°)", max_effort]
                cells.append(row)
    
            # Tracé des essais et du MAT306 dans le fig_comp
            f_max_essai, d_max_essai, f_max_simu, d_max_simu = trace_essais_comp(
                df_essai,
                essai_meta["nom_essai"],
                facteur,
                essai_meta["axe"],
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
            print(f"[ERROR] Problème rencontré lors du traitement de {file_path}: {e}")

    # S'il y a des données valides, on plot
    if not labels_list and not preview_mode:
        print("[WARNING] Aucune donnée valide trouvée pour générer les graphiques finaux.")
        return

    if not preview_mode:
        # Plot correlations and final table
        trace_global(axes_corr, labels_list, x_efforts_list, y_efforts_list, x_disp_list, y_disp_list)
        faire_tableau_recap(labels_list, x_efforts_list, y_efforts_list, x_disp_list, y_disp_list, save=save_outputs)
    
        # Styling and saving
        axe_essais.set_xlabel('Déplacement (mm)')
        axe_essais.set_ylabel("Effort (N)")
        axe_essais.set_title("Courbes d'essai de traction")
        axe_essais.legend()
    
        axe_tableau.set_axis_off()
        main_table = axe_tableau.table(
            cellText=cells, 
            colLabels=headers,
            loc="center",
            cellLoc="center"
        )
        main_table.auto_set_font_size(False)
        main_table.set_fontsize(11)
        main_table.scale(1.2, 2.2)
    
        for (r, c), cell in main_table.get_celld().items():
            cell.set_edgecolor('#E0E0E0')
            if r == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#3A6996') # Medium-light elegant blue
                
                # Increase precisely the header cell's size/height
                current_height = cell.get_height()
                if current_height is not None:
                    cell.set_height(current_height * 1.5)
            else:
                cell.set_facecolor('#F8F9FA' if r % 2 == 0 else '#FFFFFF')

    if save_outputs:
        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            fig_max.savefig(output_dir / "tableau_rivet_bleu.png", dpi=300, bbox_inches='tight')
            fig_essais.savefig(output_dir / "courbes_essais.png", dpi=300, bbox_inches='tight')
            fig_comp.savefig(output_dir / "comparaison_matis.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"[WARNING] Impossible de sauvegarder les images : {e}")

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
        
        # Intégration des graphiques polaires Mohr
        # Based on the user data structure, index 0=Tension, 1=Shear, 2=Mixed
        plot_polar_failure_criteria(
            tension_force=x_efforts_list[0], 
            shear_force=x_efforts_list[1], 
            mixed_force=x_efforts_list[2],
            save=save_outputs
        )
        
        if save_outputs:
            fig_corr.savefig(Path("output") / "correlation_efforts.png", dpi=300, bbox_inches='tight')
        
        # --- Nouveau : Tableau des paramètres de simulation ---
        # On retire h_ms pour le tableau horizontal du rapport
        table_params = {k: v for k, v in PARAMETERS_MAT306.items() if k != 'h_ms'}
        p_headers = list(table_params.keys())
        p_values = [[v for v in table_params.values()]]
        
        # Adaptation de figsize au nombre de colonnes
        width = len(p_headers)
        fig_params, ax_p = plt.subplots(figsize=(width, 2))
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
        plt.show()
    else:
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
    preview_frame = tk.LabelFrame(root, text=" 3. Aperçu des Graphes (Essais vs Modèle) ", padx=10, pady=10, fg="#3A6996", font=("Helvetica", 10, "bold"))

    def select_file(test_type, label):
        filename = filedialog.askopenfilename(
            title=f"Sélectionnez le fichier pour l'essai: {test_type}",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            paths[test_type] = filename
            label.config(text=Path(filename).name, fg="#2C3E50")

    def run_analysis(save_files=True):
        if not all(paths.values()):
            messagebox.showerror("Erreur de sélection", "Veuillez sélectionner un fichier pour CHACUN des 3 essais (Tension, Cisaillement, Mixte) avant de lancer.")
            return

        # Récupération et validation des paramètres MAT306
        try:
            for key in PARAMETERS_MAT306:
                val = entry_widgets[key].get().replace(',', '.')
                PARAMETERS_MAT306[key] = float(val)
        except ValueError:
            messagebox.showerror("Erreur de saisie", f"Paramètre invalide pour '{key}'. Veuillez entrer un nombre valide.")
            return
            
        # Sauvegarder les paramètres et les chemins pour la prochaine fois (uniquement en mode complet)
        if save_files:
            save_config(PARAMETERS_MAT306, paths)

        # Si on lance l'analyse complète, masquer l'aperçu avant d'appeler main (qui bloque)
        if save_files:
            try:
                if preview_frame.winfo_exists():
                    preview_frame.pack_forget()
            except Exception:
                pass

        # Lancer l'analyse (mode preview si save_files est False)
        fig = main(paths["tension"], paths["shear"], paths["mixed"], save_outputs=save_files, preview_mode=not save_files)

        # Si on est en mode preview et que la fenêtre Tkinter existe encore
        try:
            if not save_files and fig is not None and preview_frame.winfo_exists():
                # Nettoyer les anciens widgets de l'aperçu
                for widget in preview_frame.winfo_children():
                    widget.destroy()
                
                # Intégrer la figure matplotlib
                canvas = FigureCanvasTkAgg(fig, master=preview_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        except Exception:
            pass

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
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)

    btn_preview = tk.Button(btn_frame, text="Aperçu Rapide", font=("Helvetica", 12, "bold"), bg="#607D8B", fg="white", command=lambda: run_analysis(save_files=False), height=2, width=15)
    btn_preview.pack(side="left", padx=10)

    btn_run = tk.Button(btn_frame, text="Lancer l'Analyse", font=("Helvetica", 12, "bold"), bg="#4A90E2", fg="white", command=lambda: run_analysis(save_files=True), height=2, width=20)
    btn_run.pack(side="left", padx=10)

    root.mainloop()

if __name__ == "__main__":
    _, initial_paths = load_config()
    launch_gui(initial_paths)
