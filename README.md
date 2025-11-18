# Diffusion rapide en 1D – Schéma volumes finis

Ce dépôt contient une implémentation simple d’un schéma volumes finis implicite (maillage uniforme, conditions de Neumann) pour l’équation de diffusion rapide avec β = 1/2.

Le code calcule et trace les entropies discrètes suivantes en fonction du temps, pour différentes valeurs de α : 

- log(E_d^α[u(t)])
- log(F_d^α[u(t)])

Ces graphes reproduisent le comportement observé dans la “Figure 6” de l’article de Chainais–Jüngel–Schuchnigg (2015).

## Points clés
- Schéma implicite (Euler arrière) en temps

- Volumes finis en espace (1D, maillage uniforme)

- Résolution par méthode de Newton amortie

- Clamp pour garantir la non-négativité de la solution

- Calcul des entropies discrètes E_d^α et F_d^α

- Paramètres utilisés : α ∈ {0.5, 1, 2, 6}, β = 1/2

## Lancer la simulation
```bash
python fast_diffusion_entropy_demo.py
```

## Objectif du projet

L’objectif est de valider numériquement les propriétés théoriques des schémas volumes finis dans le cadre de la diffusion non linéaire. 
Les résultats pourront ensuite être comparés ou étendus à des modèles plus complexes (dimensions supérieures, autres régimes de diffusion, modèles tumoraux, etc.).
