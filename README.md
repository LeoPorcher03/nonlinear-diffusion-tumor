# Diffusion rapide en 1D — Schéma par volumes finis

Ce dépôt contient une implémentation numérique d’un schéma par **volumes finis implicite** (Euler arrière en temps, maillage uniforme, conditions de Neumann) pour l’équation de diffusion rapide en dimension 1 :

$$
u_t = (u^\beta)_{xx},
\quad \text{avec } \beta = \frac{1}{2}.
$$

Le but principal de ce projet est de vérifier numériquement certaines propriétés théoriques du schéma à l’aide de quantités de type entropie.

---

## Entropies discrètes

Le programme calcule et trace l’évolution en temps des entropies discrètes suivantes pour différentes valeurs du paramètre $\alpha$ :

- $\log\!\left(E_d^\alpha[u(t)]\right)$  
- $\log\!\left(F_d^\alpha[u(t)]\right)$  

Ces courbes sont conçues pour reproduire qualitativement le comportement observé dans la **Figure 6** de l’article de référence.

---

## Fonctionnalités principales

- Schéma implicite (Euler arrière) en temps  
- Discrétisation par volumes finis en espace (1D, maillage uniforme)  
- Résolution du système non linéaire par méthode de Newton amortie  
- Projection sur $\mathbb{R}^+$ pour garantir la positivité de la solution  
- Calcul et visualisation des entropies discrètes $E_d^\alpha$ et $F_d^\alpha$  
- Étude paramétrique en fonction de $\alpha$

**Paramètres utilisés :**

$$
\alpha \in \{0.5,\; 1,\; 2,\; 6\},
\quad
\beta = \frac{1}{2}.
$$

---

## Objectif du projet

L’objectif est de :

- valider numériquement les propriétés de dissipation d’entropie,  
- illustrer la stabilité du schéma,  
- vérifier la cohérence avec des résultats théoriques connus,  
- fournir une base simple et reproductible pour l’étude de régimes de diffusion non linéaire.

Ce travail constitue également une base pour des extensions futures :

- dimension 2D ou 3D,  
- autres valeurs de $\beta$,  
- modèles physiques ou biologiques (par exemple diffusion tumorale),  
- schémas numériques alternatifs.

---

## Référence

Chainais, Jüngel, Schuchnigg (2015) — *Entropy structure and convergence of finite volume schemes for nonlinear diffusion equations*.
