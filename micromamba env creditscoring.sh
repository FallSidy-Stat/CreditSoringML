micromamba activate credit-scoring

# 1. Configure Kaggle (si pas déjà fait)
mkdir -p ~/.kaggle
# Va sur kaggle.com -> ton profil -> Account -> Create New API Token
# Ça télécharge kaggle.json
# Déplace-le :
# mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Teste Kaggle
kaggle datasets list | head
# doit afficher des datasets

# 3. Lance le projet depuis le début
python src/data/ingestion.py