# 🇲🇦 Déterminants de l'Inflation au Maroc — Modèle BSVAR

**Mots-clés :** BSVAR, Inflation, Politique Monétaire, Chocs Climatiques, Bayésien, Gibbs Sampler, Sign Restrictions, Maroc

## 📌 Contexte du projet

Ce projet, réalisé dans le cadre de ma formation d'ingénieur à l'**INSEA** (Statistique, Économie Appliquée & Big Data), propose une modélisation économétrique avancée pour quantifier les sources de l'inflation au Maroc sur la période **2017–2026**.

L'objectif principal est de départager la part de l'inflation due aux **chocs climatiques** (stress hydrique/sécheresse) de celle due aux **chocs de demande** ou à la **politique monétaire**. L'originalité de l'approche réside dans l'application d'un modèle **VAR Structurel Bayésien** (BSVAR) avec restrictions de signes, permettant une identification causale robuste des chocs.

## 🔬 Approche Méthodologique

- **Modèle :** VAR(2) Bayésien avec prior de Minnesota.
- **Variables (5) :** Précipitations (climat), Inflation (IPC), Taux Directeur (TMP), Production industrielle (IPI), Taux de change effectif réel (TCER).
- **Échantillon :** Mensuel, Jan 2017 – Avr 2026 (112 obs).
- **Inférence :** Gibbs Sampler (2000 draws, 500 burn-in).
- **Identification Structurelle :** 
  - Restrictions de signes (Uhlig, 2005) pour les chocs (Climatique, Demande, Monétaire, Offre, Change).
  - Contrainte d'exogénéité par blocs : les précipitations ne répondent pas contemporainement aux chocs macroéconomiques.
- **Livrables :** Fonctions de Réponse (IRF), Décomposition de la Variance (FEVD), Décomposition Historique.

## 📊 Principaux Résultats

- Le **choc climatique** explique environ **25,5 %** de la variance de l'inflation à long terme.
- Le **choc monétaire pur** n'en explique qu'environ **18,4 %**.
- Une hausse du taux directeur (politique restrictive) réduit significativement l'inflation, mais avec un délai de 4 à 6 mois.
- **Conclusion centrale :** La politique monétaire est limitée face à une inflation d'origine climatique ; les politiques budgétaires (subventions, réserves stratégiques) et structurelles (résilience hydrique) sont indispensables en complément.

## 🛠️ Installation et Exécution

**1. Cloner le dépôt**
```bash
git clone https://github.com/kartiboucakir-beep/BSVAR_Morocco_Inflation.git
cd BSVAR_Morocco_Inflation
