# Sphère Motifs

Un simulateur de propagation en 3D qui extrait une référence depuis deux exemples et la réutilise pour retrouver une forme dans une nouvelle scène.

![Deux vues de la pièce forment un motif commun. Le motif retrouve la pièce retournée parmi un engrenage, une équerre, une rondelle et un boulon.](outputs/sphere-piece-localisee.png)

## Ce que fait la sphère

Deux motifs entrent dans le réseau sous forme d'impulsions. Une jonction compare leurs alignements et émet un signal **C** qui encode ce qu'ils ont en commun. C circule dans la sphère. À son retour, la jonction compare cette référence à une nouvelle scène. Les sondes mesurent ce qui a circulé et où la correspondance apparaît.

Nous fournissons les règles de comparaison et l'encodage des entrées. Les deux exemples déterminent la référence. Les essais de coupure permettent de vérifier si la reconnaissance dépend effectivement du retour de C.

## Résultats reproduits

| Essai | Résultat observé | Mesures |
| --- | --- | --- |
| Un L et un T dans la même scène | Deux références formées à partir d'exemples, puis deux formes retrouvées après déplacement et rotation. Le leurre obtient 4 cases sur 5. | [L et T](outputs/sphere-deux-references-mesures.json) |
| Pièce mécanique en T | Sur une troisième image synthétique, la pièce retournée parmi d'autres objets obtient 41 cases sur les 42 du motif. Le second candidat en obtient 37. | [Pièce en T](outputs/sphere-mecanique-mesures.json) |
| Boulons et écrous | Trois appariements retrouvés parmi 18 dans une image synthétique. Les écrous leurres sont presque à la même distance des boulons. Couper le retour de C supprime les trois appariements. | [Boulons en diagonale](outputs/sphere-assemblages-diagonaux-mesures.json) |

Le test de la pièce en T utilise un masque de 25 × 25 et des rotations par quarts de tour. Les essais boulon–écrou extraient les silhouettes et les axes depuis les pixels, puis comparent les positions relatives. Ces résultats concernent des images générées, vues de dessus, avec des pièces séparées et un fond contrasté.

## Rejouer les expériences

Avec Node.js 24 et Bash, depuis la racine du dépôt :

```sh
bash scripts/rejouer.sh
```

Le script recalcule les 13 rapports dans un dossier temporaire et les compare octet pour octet aux mesures versionnées. Les simulations utilisent uniquement les modules intégrés à Node.js. Les masques, les images d'entrée et les paramètres sont versionnés. Les scripts de rendu des figures utilisent Python et sont facultatifs pour rejouer les mesures.

- [Sources et paramètres](work/)
- [Images et mesures](outputs/)
- [Script de rejeu](scripts/rejouer.sh)
