# Publication du dépôt

Depuis Ubuntu / WSL, dans `neuromorphic-avoidance` :

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Add event-based robot navigation with SNN and CNN"
```

Créer sur GitHub un dépôt vide `neuromorphic-avoidance`, sans README, licence
ou .gitignore supplémentaires, puis remplacer `TON_PSEUDO` :

```bash
git remote add origin https://github.com/TON_PSEUDO/neuromorphic-avoidance.git
git push -u origin main
```

À publier : code, sources ROS, dépendances verrouillées, tests, licences, deux
petits modèles entraînés et images du README. Les modèles permettent de lancer
la démo sans recréer le dataset ni entraîner les réseaux.

Le `.gitignore` exclut les données et sorties `artifacts/`, les rapports
`reports/`, l'historique de développement local, les environnements Python,
les caches et les compilations ROS. Aucun de ces fichiers n'a été supprimé.

Contrôle facultatif avant le commit :

```bash
.venv/bin/python scripts/check_export.py
```

Ce contrôle copie seulement les fichiers candidats Git dans un dossier temporaire
et lance la simulation avec les poids inclus. Il réutilise les dépendances
installées et ne valide pas une installation neuve.
