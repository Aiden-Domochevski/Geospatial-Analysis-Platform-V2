"""Remove extra screenshots, keep only 10 best ones with detailed captions."""
import re

tex_path = r"..\documentation\rapport.tex"
with open(tex_path, "r", encoding="utf-8") as f:
    content = f.read()

# Labels to REMOVE (keep 10, remove 18)
remove_labels = [
    "screenshot_map_zoom", "screenshot_map_sidebar",
    "screenshot_rankings_bottom",
    "screenshot_clusters_radar", "screenshot_clusters_bar",
    "screenshot_mergers_table",
    "screenshot_cities_detail",
    "screenshot_deepdive_scores", "screenshot_deepdive_map",
    "screenshot_data",
    "screenshot_sitesel_map",
    "screenshot_runsite_config", "screenshot_runsite_progress",
    "screenshot_bcg_config", "screenshot_bcg_actions",
    "screenshot_enrichment_status", "screenshot_enrichment_pipeline",
    "screenshot_enrichment_console",
]

# Remove figure blocks containing these labels
for label in remove_labels:
    pattern = r'\\begin\{figure\}\[h\]\s*\\centering\s*\\fbox\{\\parbox\{[^}]*\}\{[^}]*' + re.escape(label).replace(re.escape("_"), r"[_\\].*?") + r'.*?\\end\{figure\}'
    # Simpler approach: find the figure block containing the label
    search = f"\\label{{fig:{label}}}"
    if search in content:
        # Find the begin{figure} before and end{figure} after
        idx = content.index(search)
        # Search backwards for \begin{figure}
        begin_idx = content.rfind("\\begin{figure}", 0, idx)
        # Search forwards for \end{figure}
        end_idx = content.index("\\end{figure}", idx) + len("\\end{figure}")
        # Remove the block plus any trailing newlines
        while end_idx < len(content) and content[end_idx] in '\r\n':
            end_idx += 1
        content = content[:begin_idx] + content[end_idx:]

# Now update the captions of the 10 remaining ones to be more descriptive
replacements = {
    "fig:screenshot_map": (
        "Module Map : Carte interactive Folium affichant les 1\\,198 agences du réseau Attijariwafa Bank avec code couleur par action stratégique.",
        "Module Map --- Vue d'ensemble du réseau bancaire national. Chaque marqueur représente une agence, coloré selon l'action stratégique recommandée : vert (KEEP), jaune (REVIEW), rouge (CLOSE). Le survol d'un marqueur affiche un popup HTML avec le nom, le score composite, le cluster K-Means et l'action. Le panneau latéral permet de filtrer par ville, cluster et plage de score."
    ),
    "fig:screenshot_rankings_top": (
        "Module Rankings — Vue principale : KPIs stratégiques et classement horizontal des agences par score composite.",
        "Module Rankings --- En haut, trois métriques KPI résument l'état du réseau (nombre d'agences KEEP, CLOSE, et score moyen). En dessous, un graphique à barres horizontales Plotly classe les 20 meilleures agences par score composite décroissant, permettant d'identifier immédiatement les points forts du réseau."
    ),
    "fig:screenshot_clusters_pca": (
        "Module ML Clusters — PCA 2D : Projection des agences dans l'espace des composantes principales, colorées par profil spatial.",
        "Module ML Clusters --- Projection PCA bidimensionnelle des 1\\,198 agences dans l'espace des deux premières composantes principales (72.5\\% et 10.1\\% de variance). Chaque point est coloré selon son cluster K-Means (High Performer, Commercial Hub, Residential Anchor, Underperformer, At Risk). Les diagrammes radar en dessous détaillent la signature spatiale moyenne de chaque profil."
    ),
    "fig:screenshot_mergers_map": (
        "Module Mergers — Carte de cannibalisation : Visualisation géographique des paires reliées par des segments rouges.",
        "Module Mergers --- Carte Folium dédiée à la détection de cannibalisation. Les paires d'agences distantes de moins de 1.2 km sont reliées par des segments rouges. Le tableau associé liste chaque paire avec les noms, scores composites, distance exacte en km, et la recommandation (conserver la plus performante, fusionner ou fermer la moins rentable)."
    ),
    "fig:screenshot_cities_chart": (
        "Module Cities — Comparaison : Classement des agglomérations par score composite moyen.",
        "Module Cities --- Analyse macroscopique ville par ville. Le graphique à barres compare les agglomérations par score composite moyen, nombre d'agences et proportion KEEP vs CLOSE. Un clic sur une ville affiche le détail individuel de chacune de ses agences avec leurs métriques spatiales."
    ),
    "fig:screenshot_deepdive_gauge": (
        "Module Deep Dive — Jauge : Indicateur visuel du score composite de l'agence sélectionnée avec comparaison à la moyenne du réseau.",
        "Module Deep Dive --- Fiche d'audit 360° d'une agence individuelle. La jauge circulaire Plotly affiche le score composite avec des zones colorées (rouge $< 0.20$, jaune $0.20-0.35$, vert $> 0.35$) et un delta par rapport à la moyenne du réseau. Les 6 scores spatiaux individuels sont présentés sous forme de KPI, accompagnés d'une mini-carte Folium centrée sur l'agence avec un cercle de rayon 1 km."
    ),
    "fig:screenshot_sitesel_table": (
        "Module Site Selection — Résultats : Liste des sites candidats classés par similarité avec le profil High Performer.",
        "Module Site Selection --- Tableau des sites candidats évalués par benchmarking KNN. Chaque candidat est classé par similarité cosinus avec le centroïde du cluster High Performer. Le tableau affiche le score de similarité, les 6 scores spatiaux du candidat, et l'estimation de revenu attendu calculée par vote des 5 agences existantes les plus proches en profil."
    ),
    "fig:screenshot_runsite_results": (
        "Module Run Site Selection — Résultats : Carte interactive des sites recommandés avec tableau récapitulatif exportable.",
        "Module Run Site Selection --- Résultats du moteur de découverte interactif. Après paramétrage des contraintes (distance minimum, densité, nombre de clusters) via les sliders, le pipeline à 7 étapes s'exécute automatiquement. Les Top N candidats sont affichés sur une carte ScatterMapbox avec tooltips de score, accompagnés d'un tableau récapitulatif exportable en CSV."
    ),
    "fig:screenshot_bcg_scatter": (
        "Module Network Optimization — Matrice BCG : Visualisation interactive du positionnement stratégique de chaque agence dans l'espace Potentiel vs Performance.",
        "Module Network Optimization --- Matrice BCG spatiale interactive. Chaque point représente une agence positionnée dans l'espace (Potentiel Prédit par XGBoost en abscisse, Revenu Réel en ordonnée). Les lignes médianes divisent le plan en 4 quadrants (FLAGSHIP, REVIEW/INVEST, MONITOR, CLOSE). La diagonale noire ($y=x$) représente la ligne d'espérance de potentiel. Le tableau d'actions en dessous liste les recommandations stratégiques filtrées par quadrant."
    ),
}

# No 10th — let me add one for Data Enrichment by renaming the existing one
# Actually we removed all 3 enrichment screenshots. Let me add one back.
# Find the Data Enrichment subsection and add a single figure
enrichment_marker = "\\subsection{Module 11"
if enrichment_marker in content:
    # Find the end of the controls list
    controls_end = "\\end{itemize}\n"
    idx = content.index(enrichment_marker)
    # Find the next \section after the enrichment module
    next_section = content.index("\\section{Valeur", idx)
    # Check if there are any figures left in that range
    enrichment_section = content[idx:next_section]
    if "begin{figure}" not in enrichment_section:
        # Add a figure before the next section
        new_fig = """
\\begin{figure}[h]
    \\centering
    \\fbox{\\parbox{0.9\\textwidth}{\\centering\\vspace{3cm}\\textit{Capture d'écran : Console de scraping live et pipeline Clean \\& Merge.}\\vspace{3cm}}}
    \\caption{Module Data Enrichment --- Interface de scraping Google Maps en direct. L'utilisateur choisit le mode (reprise ou nouveau), lance le scraper via le bouton Start, et observe la progression en temps réel dans une console simulée. Les métriques en haut affichent le nombre d'entrées checkpointées et de POI scrapés. Le bouton Clean \\& Merge déclenche séquentiellement le nettoyage et la fusion des données dans le fichier \\texttt{merged\\_data\\_clean.csv}.}
    \\label{fig:screenshot_enrichment}
\\end{figure}

"""
        content = content[:next_section] + new_fig + content[next_section:]

# Apply caption replacements
for label, (old_caption, new_caption) in replacements.items():
    if old_caption in content:
        content = content.replace(old_caption, new_caption)

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(content)

# Count remaining
print(f"Screenshot placeholders remaining: {content.count('fbox')}")
print(f"Total figures: {content.count('begin{figure}')}")
print("Done!")
