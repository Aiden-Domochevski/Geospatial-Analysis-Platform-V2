# Part 1: Introduction, Outils, Architecture globale, Premiers modules
PART1 = r"""
\chapter{Réalisation de la Solution}

\section{Introduction}

Ce cinquième chapitre présente la réalisation concrète de la plateforme \textit{Branch Intelligence}. Après avoir défini les fondements théoriques (Chapitre 3) et détaillé l'implémentation algorithmique (Chapitre 4), nous décrivons ici l'application web interactive qui constitue le livrable final du projet. Cette application transforme les modèles de Machine Learning et les algorithmes spatiaux en un outil d'aide à la décision directement exploitable par les décideurs d'Attijariwafa Bank, sans aucune compétence technique requise de leur part.

L'application se distingue par son architecture monolithique unifiée : l'ensemble du pipeline --- de l'acquisition des données brutes jusqu'à la recommandation stratégique finale --- est encapsulé dans un seul point d'entrée, offrant une expérience utilisateur fluide et cohérente.

\section{Outils et Technologies Utilisés}

La réalisation de la solution repose sur un écosystème technologique Python riche et mature, sélectionné pour sa robustesse, sa compatibilité et son adoption massive dans l'industrie du Machine Learning.

\subsection{Streamlit — Framework d'Interface Web}

\texttt{Streamlit} (version 1.x) est un framework Python open-source développé par Snowflake, devenu le standard de l'industrie pour le déploiement rapide d'applications de Data Science. Contrairement aux frameworks web classiques (Django, Flask) qui exigent la construction séparée d'un backend API et d'un frontend JavaScript, Streamlit permet de générer une interface web interactive complète en Python pur.

\textbf{Justification du choix :} Notre application intègre directement des modèles scikit-learn, XGBoost et des structures spatiales cKDTree dans la boucle de rendu. Streamlit est le seul framework permettant cette intégration native sans couche d'abstraction API, réduisant drastiquement le temps de développement et la complexité architecturale.

\textbf{Fonctionnalités exploitées :}
\begin{itemize}
    \item \texttt{st.tabs()} : Organisation de l'interface en 11 onglets thématiques.
    \item \texttt{@st.cache\_data} : Mémoisation du chargement CSV pour des performances instantanées.
    \item \texttt{st.session\_state} : Persistance de l'état utilisateur entre les interactions.
    \item \texttt{st.columns()}, \texttt{st.metric()} : Mise en page responsive avec KPIs visuels.
    \item \texttt{st.plotly\_chart()} : Intégration native des graphiques interactifs.
\end{itemize}

\subsection{Folium — Cartographie Interactive}

\texttt{Folium} est une bibliothèque Python qui génère des cartes interactives basées sur \texttt{Leaflet.js}. Dans notre application, Folium est utilisé pour le rendu géospatial de la carte principale (Tab 1), affichant les 1\,198 agences avec un code couleur correspondant à leur action stratégique recommandée (KEEP, REVIEW, CLOSE). Chaque marqueur est enrichi d'un popup HTML contenant le nom de l'agence, son score composite, son cluster K-Means et son action.

\subsection{Plotly — Visualisation Interactive Avancée}

\texttt{Plotly} est la bibliothèque de visualisation principale de l'application. Contrairement à Matplotlib (statique), Plotly génère des graphiques interactifs permettant le zoom, le survol (hover tooltips), et l'export PNG. Les types de graphiques exploités incluent :
\begin{itemize}
    \item \texttt{Scatter} : Nuages de points pour la Matrice BCG et les projections PCA.
    \item \texttt{Bar} : Classements horizontaux des agences et comparaisons par ville.
    \item \texttt{Indicator / Gauge} : Jauges circulaires pour le score composite dans le module Deep Dive.
    \item \texttt{Heatmap} : Matrices de corrélation et diagrammes radar pour le profilage des clusters.
    \item \texttt{Choropleth / ScatterMapbox} : Cartes géographiques intégrées pour le module Site Selection.
\end{itemize}

\subsection{Scikit-learn — Pipeline de Machine Learning}

\texttt{Scikit-learn} fournit l'infrastructure complète du pipeline ML :
\begin{itemize}
    \item \texttt{StandardScaler} : Normalisation Z-Score préalable au clustering.
    \item \texttt{KMeans} ($K=5$, $n\_init=10$) : Segmentation non supervisée du réseau.
    \item \texttt{DBSCAN} : Détection de zones denses organiques.
    \item \texttt{PCA} : Réduction de dimensionnalité pour la visualisation 2D/3D des clusters.
\end{itemize}

\subsection{XGBoost — Modèle de Prédiction}

\texttt{XGBoost} (eXtreme Gradient Boosting) est utilisé dans le module Network Optimization (Tab 10) pour la prédiction des revenus basée sur les caractéristiques spatiales. Le modèle entraîné en temps réel dans le dashboard (\texttt{xgb.XGBRegressor(n\_estimators=100)}) produit les prédictions qui alimentent la matrice BCG.

\subsection{SciPy — Algorithmique Spatiale}

Le module \texttt{scipy.spatial.cKDTree} (implémenté en C++) est le moteur d'indexation spatiale de l'application. Il est utilisé dans trois contextes critiques :
\begin{itemize}
    \item Détection des fusions (Tab 4 - Mergers) via \texttt{query\_pairs(r=0.012)}.
    \item Enrichissement des candidats en densité de population (Tab 9 - Run Site Selection).
    \item Identification des agences proches dans le module d'optimisation réseau (Tab 10).
\end{itemize}

\subsection{Selenium WebDriver — Scraping Dynamique}

\texttt{Selenium} pilote un navigateur Chrome en mode \textit{headless} pour le scraping Google Maps. L'intégration dans le dashboard (Tab 11) utilise \texttt{subprocess.Popen} et \texttt{threading.Thread} pour une exécution asynchrone non-bloquante.

\subsection{Pandas et NumPy — Manipulation des Données}

\texttt{Pandas} constitue l'épine dorsale de toutes les transformations de données : chargement CSV, filtrage vectorisé, agrégation par groupe (\texttt{groupby}), et jointures. \texttt{NumPy} fournit les opérations matricielles sous-jacentes pour les calculs de distance et de similarité cosinus.

\subsection{Synthèse de l'Environnement Technique}

\begin{table}[h]
\centering
\begin{tabular}{|l|l|l|}
\hline
\textbf{Catégorie} & \textbf{Technologie} & \textbf{Usage} \\ \hline
Interface Web & Streamlit & Framework principal du dashboard \\ \hline
Cartographie & Folium + Leaflet.js & Carte interactive (Tab 1) \\ \hline
Visualisation & Plotly & Graphiques interactifs (11 tabs) \\ \hline
Machine Learning & Scikit-learn, XGBoost & K-Means, PCA, Régression \\ \hline
Spatial & SciPy (cKDTree) & Requêtes spatiales $O(N \log N)$ \\ \hline
Scraping & Selenium + Chrome & Acquisition Google Maps \\ \hline
Data & Pandas, NumPy & ETL et calculs matriciels \\ \hline
Langage & Python 3.14 & Langage principal \\ \hline
\end{tabular}
\caption{Synthèse de la stack technologique de la plateforme Branch Intelligence.}
\label{tab:tech_stack}
\end{table}

\section{Architecture de l'Application}

L'application est structurée en \textbf{11 onglets (Tabs)} organisés selon une progression logique : de l'exploration générale vers l'action spécifique. Chaque onglet constitue un module fonctionnel autonome mais interconnecté aux autres via le dataset partagé en mémoire. Les sections suivantes détaillent chaque module.

\subsection{Module 1 — Carte Interactive (Map)}

Le premier onglet constitue le point d'entrée visuel de l'application. Il présente une carte interactive \texttt{Folium} centrée sur le Maroc, sur laquelle les 1\,198 agences sont représentées par des marqueurs circulaires dont la couleur encode l'action stratégique recommandée :

\begin{itemize}
    \item \textbf{Vert} (\texttt{\#22C55E}) : KEEP — Invest (agences performantes à maintenir).
    \item \textbf{Vert clair} (\texttt{\#86EFAC}) : KEEP — Flagship (performantes mais proches d'une autre agence).
    \item \textbf{Jaune} (\texttt{\#EAB308}) : REVIEW — À auditer.
    \item \textbf{Orange} (\texttt{\#F97316}) : REVIEW — Candidate à une fusion.
    \item \textbf{Rouge} (\texttt{\#EF4444}) : CLOSE — Redondante.
    \item \textbf{Rouge foncé} (\texttt{\#991B1B}) : CLOSE — Sous-performante.
\end{itemize}

Chaque marqueur est enrichi d'un popup HTML affichant : le nom de l'agence, le score composite, le cluster K-Means, et l'action recommandée. Un panneau latéral permet de filtrer par ville, par cluster, et par plage de score composite.

\begin{figure}[h]
    \centering
    \fbox{\parbox{0.9\textwidth}{\centering\vspace{3cm}\textit{Capture d'écran : Vue globale de la carte interactive avec les marqueurs colorés par action stratégique.}\vspace{3cm}}}
    \caption{Module Map : Carte interactive Folium affichant les 1\,198 agences du réseau Attijariwafa Bank avec code couleur par action stratégique.}
    \label{fig:screenshot_map}
\end{figure}

\subsection{Module 2 — Classements (Rankings)}

Ce module présente les agences sous forme de classements tabulaires dynamiques. L'utilisateur peut trier par score composite (Top 20, Bottom 20), par ville, ou par action. Un graphique à barres horizontales Plotly accompagne le tableau, permettant une lecture visuelle immédiate de la hiérarchie du réseau.

Trois KPIs sont affichés en en-tête :
\begin{itemize}
    \item Nombre d'agences à conserver (KEEP).
    \item Nombre d'agences à fermer (CLOSE).
    \item Score composite moyen du réseau.
\end{itemize}

\begin{figure}[h]
    \centering
    \fbox{\parbox{0.9\textwidth}{\centering\vspace{3cm}\textit{Capture d'écran : Classement des agences par score composite avec graphique à barres.}\vspace{3cm}}}
    \caption{Module Rankings : Vue tabulaire interactive avec classement et métriques KPI.}
    \label{fig:screenshot_rankings}
\end{figure}

\subsection{Module 3 — Clusters ML}

Ce module exploite directement les résultats du clustering K-Means ($K=5$) et de la projection PCA présentés au Chapitre 4. L'interface se compose de :

\begin{enumerate}
    \item \textbf{Scatter Plot PCA 2D :} Projection des agences dans l'espace des deux premières composantes principales (72.5\% + 10.1\% de variance), colorées par cluster. L'utilisateur peut survoler chaque point pour obtenir les détails de l'agence.
    \item \textbf{Radar Charts par Cluster :} Diagrammes radar affichant la signature spatiale moyenne de chaque profil (Commercial, Office, Residential, Accessibility, Competition, Cannibalization).
    \item \textbf{Répartition par Cluster :} Diagramme en barres montrant le nombre d'agences dans chaque profil.
\end{enumerate}

\begin{figure}[h]
    \centering
    \fbox{\parbox{0.9\textwidth}{\centering\vspace{3cm}\textit{Capture d'écran : Projection PCA 2D des 5 clusters avec radar charts.}\vspace{3cm}}}
    \caption{Module ML Clusters : Visualisation interactive des 5 profils d'agences identifiés par K-Means.}
    \label{fig:screenshot_clusters}
\end{figure}
"""

with open("ch5_part1.tex", "w", encoding="utf-8") as f:
    f.write(PART1)
print("Part 1 written successfully.")
