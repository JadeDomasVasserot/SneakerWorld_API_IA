# IA-API.py
import os, json, time, base64
from io import BytesIO
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
import torch

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from diffusers import DiffusionPipeline

# -----------------------------
# Config générale
# -----------------------------
os.environ.setdefault("OMP_NUM_THREADS", "4")
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ART = os.getenv("ART_DIR", "ml_artifacts")

# === 📦 Configuration Base de Données ===
DB_USER = "sneaker"
DB_PASSWORD = quote_plus('$ne@kerW0rld')
DB_HOST = "bdd_pfe_sneakerworld.serverjadedomasvasserot.com"
DB_PORT = "39168"
DB_NAME = "sneakerworld"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# === 🚀 Chargement modèles ML ===
# Recommandation embeddings
CATALOG_PATH = os.getenv("CATALOG_PATH", "sneakers.csv")
baskets = pd.read_csv(CATALOG_PATH)
basket_embeddings = np.load("sneaker_embeddings.npy")
baskets['brand'] = baskets['text'].str.split().str[0]

from sentence_transformers import SentenceTransformer
model_recommend = SentenceTransformer('all-MiniLM-L6-v2')

# Génération prompt & image
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

pipe = DiffusionPipeline.from_pretrained("Lykon/DreamShaper")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# === 📋 Schemas API ===
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 3

# === 🧩 Fonctions utilitaires ===
def get_user_preferences(user_id: int) -> List[str]:
    # Requête interactions
    interaction_query = f"""
        SELECT
            a.action_user_name AS action,
            s.sneaker_brand,
            s.sneaker_name_complete,
            s.sneaker_color,
            s.sneaker_category_occasion,
            s.sneaker_upper_material,
            s.sneaker_sole_material,
            s.sneaker_closure,
            s.sneaker_gender,
            s.sneaker_season
        FROM interaction i
        JOIN sneaker s ON i.interaction_sneaker_id = s.sneaker_model_numero
        JOIN action_user a ON i.interaction_action_id = a.action_user_id
        WHERE i.interaction_user_id = {user_id}
        ORDER BY RAND()
        LIMIT 3000;
    """
    interactions_df = pd.read_sql(interaction_query, engine)

    def format_row_with_score(row):
        action_scores = {
            "superlike": 2,
            "like": 1,
            "dislike": 0,
            "superdislike": -1
        }
        score = action_scores.get(row["action"].lower(), None)
        prefix_map = {
            2: "✅✅",
            1: "✅",
            0: "❌",
            -1: "❌❌"
        }
        prefix = prefix_map.get(score, "❓")

        return (f"{prefix} (score={score}) {row['sneaker_brand']} {row['sneaker_name_complete']} {row['sneaker_color']} – "
                f"style {row['sneaker_category_occasion']}, matière {row['sneaker_upper_material']}, "
                f"semelle {row['sneaker_sole_material']}, fermeture {row['sneaker_closure']}, "
                f"genre {row['sneaker_gender']}, saison {row['sneaker_season']}")


    prefs = [format_row_with_score(row) for _, row in interactions_df.iterrows()]

    # Requête profil user
    user_query = f"""
        SELECT user_brand, user_category, user_color, user_genre
        FROM user
        WHERE user_id = {user_id};
    """
    user_df = pd.read_sql(user_query, engine)
    profile = user_df.iloc[0] if not user_df.empty else None

    if profile is not None:
        if profile.user_brand:
            prefs.append(f"✅ préfère la marque {profile.user_brand}")
        if profile.user_category:
            prefs.append(f"✅ aime les baskets pour {profile.user_category}")
        if profile.user_color:
            prefs.append(f"✅ préfère les couleurs {profile.user_color}")
        if profile.user_genre:
            prefs.append(f"✅ genre ciblé : {profile.user_genre}")

    return prefs[:100]

def build_prompt_generate_image(user_id: int) -> str:
    prefs = get_user_preferences(user_id)
    if not prefs:
        raise HTTPException(status_code=404, detail="Aucune préférence trouvée pour cet utilisateur.")
    instruction = (
        "\n\n### INSTRUCTION:\n"
        "Using the preferences above, generate a single, high-quality English prompt suitable for Stable Diffusion "
        "to create an ideal sneaker. Include precise visual and contextual details such as: brand, sneaker type, "
        "materials (upper and sole), color palette, silhouette, closure system, target gender, preferred season, and "
        "intended use (e.g. running, lifestyle, basketball). Do not list attributes — write one coherent, natural prompt "
        "in English. Output only the final prompt."
    )
    return "\n".join(prefs) + instruction

# === ⏱️ Classe pour suivi génération tokens ===
class PrintAndStopCriteria(StoppingCriteria):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.counter = 0
        self.start_time = time.time()

    def __call__(self, input_ids, scores, **kwargs):
        self.counter += 1
        elapsed = time.time() - self.start_time
        print(f"⏱️ {self.counter} tokens générés en {elapsed:.2f} sec", end="\r")
        return self.counter >= self.max_tokens

# === 🚀 Création de l’API ===
app = FastAPI()

@app.post("/recommend")
def recommend_sneakers(request: RecommendationRequest):
    user_prefs = get_user_preferences(request.user_id)
    if not user_prefs:
        raise HTTPException(status_code=404, detail="Aucune interaction trouvée pour cet utilisateur.")

    likes = [p for p in user_prefs if p.startswith("✅")]
    dislikes = [p for p in user_prefs if p.startswith("❌")]

    like_vecs = model_recommend.encode(likes, batch_size=8, convert_to_numpy=True) if likes else np.zeros((1, model_recommend.get_sentence_embedding_dimension()))
    dislike_vecs = model_recommend.encode(dislikes, batch_size=8, convert_to_numpy=True) if dislikes else np.zeros((1, model_recommend.get_sentence_embedding_dimension()))

    user_vector = np.mean(like_vecs, axis=0) - np.mean(dislike_vecs, axis=0)
    user_vector = normalize(user_vector.reshape(1, -1))

    nn = NearestNeighbors(n_neighbors=50, metric="euclidean")
    nn.fit(basket_embeddings)
    distances, indices = nn.kneighbors(user_vector)

    recommended = baskets.iloc[indices[0]]
    recommended_unique = recommended.drop_duplicates(subset='brand').head(request.top_k)

    return recommended_unique.to_dict(orient="records")

@app.post("/generate_prompt_image")
def generate_prompt_image(request: RecommendationRequest):
    prompt = build_prompt_generate_image(request.user_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([PrintAndStopCriteria(500)])
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_prompt = full_text.replace(prompt, "").strip()
    print(generated_prompt)
    image = pipe(generated_prompt).images[0]
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "generated_prompt": generated_prompt,
        "image_base64": img_str
    }
