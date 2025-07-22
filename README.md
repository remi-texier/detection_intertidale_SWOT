# detection_intertidale_SWOT

Projet de création de dataset supervisé contenant les données ssh et sig0 de la mission SWOT, ainsi qu'une vérité terrain de la ligne d'eau fabriquée à partir des données topo-bathymétrique de haute précision Litto3D.

L'objectif est d'effectuer une segmentation automatique des zones intertidales à partir des données SWOT uniquement, pour cela différents modèles de deep learning sont utilisés.

---

"src" contient le programme permettant la génération du jeu de données, il faut ajouter les données brutes LR du SWOT dans le dossier "data", les données marégraphiques et les données topo-bathymétriques de Litto3D.