import os
import pandas as pd
import numpy as np
import base64
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# 1. Load environment variables
load_dotenv()

# 2. IMAGE ENCODING (The fix for the background)
def get_base64_encoded_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

bg_base64 = get_base64_encoded_image("wallpaperflare.com_wallpaper.jpg")

# 3. Data Preparation
books = pd.read_csv("books_with_emotion.csv", encoding="utf-8")
books["isbn13"] = books["isbn13"].astype(str)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"])

# 4. Vector Database Setup
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

def retrieve_semantic_recommendations(query: str, category: str = "All", tone: str = "All", initial_top_k: int = 50, final_top_k: int = 16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [rec.page_content.strip('"').split()[0] for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].copy()
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    tone_map = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
    if tone in tone_map:
        book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)
    return book_recs.head(final_top_k)

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    if recommendations.empty: return []
    results = []
    for _, row in recommendations.iterrows():
        authors_split = str(row["authors"]).split(";")
        authors_str = f"{authors_split[0]} and {authors_split[1]}" if len(authors_split) == 2 else (f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}" if len(authors_split) > 2 else str(row["authors"]))
        caption = f"**{row['title']}** by {authors_str}\n\n{' '.join(str(row['description']).split()[:30])}..."
        results.append((row["large_thumbnail"], caption))
    return results

# 5. CUSTOM CSS (Using the encoded image)
# We use an f-string here to inject the base64 string directly into the background-image property
custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=EB+Garamond:wght@400;700&display=swap');

.gradio-container {{ 
    background-image: url('data:image/jpg;base64,{bg_base64}') !important; 
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}}

.inner-box {{
    background-color: rgba(26, 18, 11, 0.9) !important; 
    border: 2px solid #AE866A !important;
    border-radius: 15px !important;
    padding: 25px !important;
    margin-bottom: 25px !important;
    backdrop-filter: blur(5px);
}}

.gr-form, .gr-box, .form {{ background-color: transparent !important; border: none !important; }}

input, textarea, .gr-dropdown, .gr-select, .gr-input, select {{
    background-color: #1a120b !important; 
    color: #D5B78B !important;            
    font-family: 'EB Garamond', serif !important;
    font-weight: 700 !important;
    border: 1px solid #AE866A !important;
}}

textarea {{ height: 180px !important; }}

.prose h1 {{
    font-family: 'Playfair Display', serif !important;
    color: #F4EBD0 !important;
    font-size: 3.5rem !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.9);
    text-align: center;
}}

label span {{
    font-family: 'EB Garamond', serif !important;
    font-weight: 700 !important;
    color: #D5B78B !important;
    text-transform: uppercase;
}}

.gr-button-primary {{
    background: rgba(87, 69, 36, 1) !important;      
    color: #D5B78B !important;           
    border: 1px solid #AE866A !important;
    border-radius: 10px !important;      
    font-family: 'EB Garamond', serif !important;
    font-weight: 700 !important;
    text-transform: uppercase;
}}
"""

library_theme = gr.themes.Soft(primary_hue="stone", font=[gr.themes.GoogleFont("EB Garamond"), "serif"])

with gr.Blocks(theme=library_theme, css=custom_css) as dashboard:
    gr.Markdown("# Bibliomania Book Recommender")

    with gr.Column(elem_classes="inner-box"):
        with gr.Row():
            with gr.Column(scale=3):
                user_query = gr.Textbox(label="Consult the Archive", placeholder="e.g., A tale of forgotten maps...", lines=7)
            with gr.Column(scale=1):
                category_dropdown = gr.Dropdown(choices=["All"] + sorted(books["simple_categories"].unique().tolist()), label="Collection", value="All")
                tone_dropdown = gr.Dropdown(choices=["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"], label="Atmosphere", value="All")
                submit_button = gr.Button("Search Records")

    with gr.Column(elem_classes="inner-box"):
        output = gr.Gallery(label="Recommended Volumes", columns=4, rows=2, height="auto")

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    # The 'allowed_paths' is still good practice, but base64 usually bypasses the need for it.
    dashboard.launch(allowed_paths=["."])