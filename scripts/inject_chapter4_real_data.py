import json
import os

metrics_path = r"metrics.json"
tex_file = r"..\documentation\rapport.tex"

with open(metrics_path, "r") as f:
    m = json.load(f)

reg = m["regression"]
clf = m["classification"]

# Create the dynamic tables
table_reg = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\textbf{{Modèle}} & \\textbf{{RMSE (MAD)}} & \\textbf{{MAE (MAD)}} & \\textbf{{$R^2$}} \\\\ \\hline
Régression Linéaire (OLS) & {reg['OLS']['RMSE']:,.2f} & {reg['OLS']['MAE']:,.2f} & {reg['OLS']['R2']:.4f} \\\\ \\hline
Régression Ridge & {reg['Ridge']['RMSE']:,.2f} & {reg['Ridge']['MAE']:,.2f} & {reg['Ridge']['R2']:.4f} \\\\ \\hline
Random Forest & \\textbf{{{reg['RandomForest']['RMSE']:,.2f}}} & \\textbf{{{reg['RandomForest']['MAE']:,.2f}}} & \\textbf{{{reg['RandomForest']['R2']:.4f}}} \\\\ \\hline
XGBoost & {reg['XGBoost']['RMSE']:,.2f} & {reg['XGBoost']['MAE']:,.2f} & {reg['XGBoost']['R2']:.4f} \\\\ \\hline
\\end{{tabular}}
\\caption{{Performances comparées des modèles de régression sur la prédiction des revenus.}}
\\label{{tab:reg_benchmark}}
\\end{{table}}
"""

table_clf = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Modèle de Classification}} & \\textbf{{Précision Globale (Accuracy)}} \\\\ \\hline
Régression Logistique & {clf['LogisticRegression']['Accuracy']:.4f} \\\\ \\hline
K-Nearest Neighbors (KNN) & {clf['KNN']['Accuracy']:.4f} \\\\ \\hline
Random Forest Classifier & {clf['RandomForestClassifier']['Accuracy']:.4f} \\\\ \\hline
XGBoost Classifier & {clf['XGBClassifier']['Accuracy']:.4f} \\\\ \\hline
\\end{{tabular}}
\\caption{{Performances comparées des modèles de classification multi-classes.}}
\\label{{tab:clf_benchmark}}
\\end{{table}}
"""

# The raw LaTeX chapter
chapter4_content = r"""
%=============================================================================%
% CHAPITRE 4 : IMPLÉMENTATION TECHNIQUE, MODÉLISATION ET BENCHMARKING
%=============================================================================%
\chapter{Implémentation Technique, Modélisation Appliquée et Benchmarking}

Ce quatrième chapitre détaille l'implémentation algorithmique de bout en bout du projet \textit{Branch Intelligence}. La complexité inhérente à l'analyse spatio-financière à l'échelle d'un pays exige une architecture modulaire et robuste. Ce chapitre décompose méthodiquement le cycle de vie de la donnée, depuis son extraction brute jusqu'à son exploitation au sein d'une interface d'aide à la décision. Nous présenterons également les résultats concrets de nos entraînements de modèles, justifiant scientifiquement nos choix technologiques.

\section{Architecture Globale et Définition des Cas d'Usage}

L'ensemble de l'architecture logicielle a été conçu pour répondre simultanément à deux impératifs stratégiques distincts, modélisés sous la forme de deux Cas d'Usage (Use Cases - UC) majeurs :

\begin{itemize}
    \item \textbf{Cas d'Usage 1 (UC1) : Évaluation et Optimisation du Réseau Existant.} Ce volet se concentre sur l'audit des agences bancaires actuellement en opération. L'objectif technique est d'utiliser le Machine Learning pour isoler la "qualité géographique" intrinsèque d'une agence de la "performance managériale" de son équipe, aboutissant à une Matrice BCG de restructuration. Il intègre également la détection spatiale des cannibalismes pour suggérer des fusions.
    \item \textbf{Cas d'Usage 2 (UC2) : Découverte et Sélection de Nouveaux Sites (Site Selection).} Ce volet est prospectif. Il s'agit de mailler l'ensemble du territoire marocain pour évaluer, via des algorithmes de recherche de plus proches voisins (K-NN) et de similarité vectorielle, les zones vierges présentant un potentiel commercial similaire aux agences les plus performantes du réseau actuel.
\end{itemize}

Bien que ces deux cas d'usage diffèrent dans leur finalité métier, ils partagent un tronc commun indispensable : l'ingénierie des données géospatiales.

%-----------------------------------------------------------------------------%
\section{Orchestration et Ingénierie des Données (Tronc Commun)}
%-----------------------------------------------------------------------------%

L'intelligence spatiale repose sur la granularité et la fraîcheur des données Points of Interest (POI). Le processus d'ingénierie se divise en trois phases critiques.

\subsection{Collecte Multi-Sources et Scraping Dynamique}

La couverture exhaustive du territoire nécessite l'agrégation de sources hétérogènes :
\begin{enumerate}
    \item \textbf{Geofabrik et OpenStreetMap (OSM) :} Extraction massive de données statiques via l'outil en ligne de commande \texttt{overturemaps}. Les requêtes ciblent un cadre englobant (Bounding Box) couvrant le Maroc (\texttt{-13.2, 27.6, -0.9, 35.9}) au format \texttt{geoparquet}, converti ensuite en CSV pour les catégories d'infrastructures lourdes (hôpitaux, universités).
    \item \textbf{Scraping Dynamique (Google Maps) :} Pour les commerces à forte volatilité (cafés, restaurants, nouvelles franchises), les bases OSM sont souvent obsolètes. Nous avons développé un scraper personnalisé utilisant \texttt{Selenium WebDriver}.
\end{enumerate}

Le scraper opère selon une heuristique de quadrillage. Le territoire est divisé en cellules de $0.3$ degrés de latitude et longitude. Pour chaque cellule et pour 30 mots-clés distincts, le script simule un navigateur Chrome (Headless), navigue sur l'URL de recherche Google Maps correspondante, et interagit avec le DOM (Document Object Model) via des injections JavaScript (\texttt{arguments[0].scrollTop += 800}) pour forcer le chargement asynchrone (AJAX) des résultats.

\subsection{Nettoyage, Encodage et Fusion (\texttt{cleanall.py})}

L'unification de ces sources disparates pose des défis stricts de standardisation. Le pipeline d'intégration exécute les opérations suivantes :
\begin{itemize}
    \item \textbf{Résolution des Encodages :} Les données issues du web présentent des corruptions de caractères (comme les octets \texttt{0xe9} pour les accents). Le pipeline force la lecture via l'encodage \texttt{latin1} puis sauvegarde le résultat final en \texttt{UTF-8} universel.
    \item \textbf{Dédoublonnage Spatial Stricte :} Une même pharmacie peut exister dans la base OSM et la base Google Maps sous des noms légèrement différents ("Pharmacie Centrale" vs "Pharm. Centrale"). Le dédoublonnage textuel étant inopérant, nous utilisons la vérité terrain géométrique. L'instruction \texttt{pandas} \texttt{drop\_duplicates(subset=['latitude', 'longitude'])} garantit l'unicité des coordonnées.
\end{itemize}

\subsection{Ingénierie des Caractéristiques Spatiales (Feature Engineering)}

Pour transformer un simple point GPS en vecteur de caractéristiques exploitable par nos algorithmes, nous appliquons une agrégation radiale. Pour chaque agence (UC1) ou candidat (UC2), nous générons des isochrones circulaires de rayons fixes (1 km et 2 km).
L'algorithme compte les densités de POI à l'intérieur de ces rayons pour créer un vecteur $X_i \in \mathbb{R}^p$. 

%-----------------------------------------------------------------------------%
\section{Cas d'Usage 1 (UC1) : Évaluation et Optimisation du Réseau Existant}
%-----------------------------------------------------------------------------%

Ce premier cas d'usage vise à auditer le réseau bancaire actuel en combinant apprentissage non supervisé et supervisé.

\subsection{Apprentissage Non Supervisé : Profilage K-Means}
Nous appliquons un algorithme K-Means pour segmenter le réseau existant. Préalablement, la matrice des scores environnementaux $X$ subit une transformation Z-Score (\texttt{StandardScaler}) pour s'assurer que la fonction de distance Euclidienne du K-Means ne soit pas dominée par les variables à forte variance numérique. L'algorithme, exécuté avec $K=5$, identifie les cinq profils spatiaux du réseau (High Performer, Commercial Hub, etc.).

\subsection{Benchmarking Algorithmique : Prédiction des Revenus (Régression)}

Pour évaluer objectivement une agence, nous devons estimer son potentiel théorique. Il s'agit d'un problème de régression où la variable cible $Y$ est le revenu mensuel, et la matrice $X$ contient les densités de POI. Afin de justifier scientifiquement notre choix de modèle, nous avons conduit un entraînement complet et une analyse comparative sur nos véritables données d'agences.

\subsubsection{Résultats Comparatifs des Modèles de Régression}
Nous avons entraîné et évalué quatre modèles : OLS, Ridge, Random Forest et XGBoost. Les résultats empiriques obtenus sur l'ensemble de test ($20\%$) sont résumés dans le tableau ci-dessous :

__REPLACE_TABLE_REG__

Comme illustré par les graphiques de dispersion (Actual vs Predicted) ci-dessous, les modèles ensemblistes arborescents (Random Forest et XGBoost) surpassent largement les modèles linéaires. La relation entre les commerces et le revenu est hautement non-linéaire. Bien que le Random Forest affiche empiriquement un $R^2$ très compétitif, XGBoost reste notre choix d'implémentation finale pour sa rapidité d'inférence en production et sa gestion mathématique avancée des valeurs aberrantes via les arbres séquentiels.

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/regression_benchmark.png}
    \caption{Nuages de points (Scatter Plots) comparant les Revenus Réels aux Revenus Prédits pour chaque modèle. La ligne rouge pointillée représente la prédiction parfaite ($y=x$).}
    \label{fig:reg_scatter}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/xgboost_importance.png}
    \caption{Importance des caractéristiques spatiales (Feature Importance) calculée par l'algorithme XGBoost. La dynamique commerciale s'avère être le facteur le plus prédictif.}
    \label{fig:feature_importance}
\end{figure}

\subsection{Benchmarking Algorithmique : Actionabilité (Classification)}

En parallèle, nous avons modélisé le problème sous forme de classification pure : prédire directement la recommandation d'action (KEEP, REVIEW, CLOSE) basée sur le positionnement du revenu par rapport à des seuils stratégiques. Les modèles de classification ont été évalués sur leur précision (Accuracy) et matrices de confusion.

__REPLACE_TABLE_CLF__

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/classification_benchmark.png}
    \caption{Matrices de Confusion pour chaque modèle de classification, détaillant les taux de Vrais Positifs et Faux Négatifs pour les décisions KEEP, REVIEW et CLOSE.}
    \label{fig:confusion_matrices}
\end{figure}

\subsection{La Matrice BCG Spatiale et l'Analyse des Écarts (Gap Analysis)}

Dans le dashboard final, c'est l'approche \textbf{Gap Analysis} (via la régression XGBoost) qui pilote la matrice BCG. L'espace bidimensionnel formé par $(X=\hat{Y_i}, Y=Y_i)$ délimite 4 quadrants stratégiques.
Le plan est divisé par deux lignes critiques :
\begin{enumerate}
    \item \textbf{La médiane des prédictions (Ligne Verticale) :} Sépare les zones pauvres en potentiel de celles riches en potentiel.
    \item \textbf{La médiane des revenus réels (Ligne Horizontale) :} Sépare la sous-performance de la forte rentabilité.
\end{enumerate}

\subsubsection{La Ligne d'Espérance de Potentiel (Diagonale de Performance)}
L'élément analytique le plus puissant est la \textbf{Ligne d'Espérance de Potentiel}, définie par l'équation linéaire $y = x$.
Si une agence se situe exactement sur cette ligne, sa rentabilité financière est parfaitement alignée avec le potentiel de sa rue. Le travail de l'équipe managériale correspond à 100\% aux attentes du modèle de Machine Learning. Toute déviation en dessous de la diagonale ($y < x$) caractérise une destruction de la valeur latente géographique (résidu négatif).

\subsection{Optimisation par Fusion : Cannibalisation via cKDTree}

Pour détecter la cannibalisation (redondance spatiale), l'utilisation d'une matrice de distances classique de taille $O(N^2)$ ralentirait l'interface.
Nous avons implémenté l'algorithme des Arbres k-d via \texttt{scipy.spatial.cKDTree}. L'arbre partitionne l'espace géographique $\mathbb{R}^2$. Le système exécute la fonction \texttt{query\_pairs(r=0.012)}, détectant en temps $O(N \log N)$ les agences distantes de moins de 1.2 km afin de suggérer des fusions d'actifs.

%-----------------------------------------------------------------------------%
\section{Cas d'Usage 2 (UC2) : Modélisation Algorithmique de "Site Selection"}
%-----------------------------------------------------------------------------%

Le second cas d'usage vise à auditer le territoire national vierge pour proposer des nouveaux emplacements d'agences bancaires optimaux.

\subsection{Algorithme K-Nearest Neighbors et Similarité Cosinus}

Une fois le territoire quadrillé et les filtres vectoriels (densité, cannibalisation) appliqués, l'algorithme identifie le top 20 des meilleurs sites nationaux via KNN. 
La mesure de similarité choisie pour évaluer un candidat n'est pas la Distance Euclidienne conventionnelle, mais la \textbf{Similarité Cosinus}.
\begin{equation}
\text{Sim}(C, B_{cible}) = \cos(\theta) = \frac{C \cdot B_{cible}}{||C||_2 ||B_{cible}||_2}
\end{equation}
La Similarité Cosinus garantit qu'un candidat situé dans un quartier moyen-dense avec le \textit{même ratio} commerce/résidence qu'une très grande agence performante sera classé comme excellent, car son orientation structurelle est identique. 

%-----------------------------------------------------------------------------%
\section{Architecture de Déploiement Logiciel (Dashboard)}
%-----------------------------------------------------------------------------%

L'entièreté de cette intelligence est hébergée sur une application monolithique de 3000+ lignes via le framework \texttt{Streamlit}. 
Pour intégrer l'onglet asynchrone d'enrichissement de données (Scraping Live), l'application implémente un système multi-thread complexe : création de processus (\texttt{subprocess.Popen}), forçage d'encodage ANSI/UTF-8 (\texttt{PYTHONIOENCODING=utf-8}), et \texttt{queue.Queue} thread-safe pour ne pas "freezer" l'interface graphique de rendu web, créant un pipeline Data End-to-End hautement industriel.
"""

chapter4_content = chapter4_content.replace("__REPLACE_TABLE_REG__", table_reg)
chapter4_content = chapter4_content.replace("__REPLACE_TABLE_CLF__", table_clf)

# Now we inject this into rapport.tex
try:
    with open(tex_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    split_marker = "%=============================================================================%\n% CHAPITRE 4 :"
    
    if split_marker in content:
        base_content = content.split(split_marker)[0]
    else:
        # Fallback if marker not found, inject before end document
        base_content = content.replace(r"\end{document}", "")
        
    final_latex = base_content + split_marker + chapter4_content + "\n\\end{document}\n"
    
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(final_latex)
        
    print("Chapter 4 updated successfully with REAL NUMBERS and METRICS!")
except Exception as e:
    print(f"Error: {e}")
