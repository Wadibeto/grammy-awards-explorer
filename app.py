import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="Grammy Awards Explorer",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appliquer un style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #C0C0C0;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-text {
        background-color: #2E2E2E;
        border-left: 5px solid #FFD700;
        padding: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2E2E2E;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFD700 !important;
        color: black !important;
    }
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.8rem;
        color: #888;
    }
    .stProgress > div > div > div > div {
        background-color: #FFD700;
    }
    .auto-eval {
        background-color: #2E2E2E;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger les données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("grammy_winners.csv")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Fonction pour prétraiter les données
def preprocess_data(df):
    # Convertir l'année en entier
    df['year'] = df['year'].astype(int)
    
    # Créer une colonne pour la décennie
    df['decade'] = df['year'].apply(lambda x: f"{(x//10)*10}s")
    
    # S'assurer que les colonnes textuelles sont des chaînes de caractères et remplacer NaN
    df['artist'] = df['artist'].fillna("Unknown").astype(str)
    df['category'] = df['category'].fillna("Other").astype(str)
    df['song_or_album'] = df['song_or_album'].fillna("").astype(str)
    
    # Extraire le nom de l'artiste principal (pour les cas avec plusieurs artistes)
    df['main_artist'] = df['artist'].apply(lambda x: x.split(',')[0].split('&')[0].strip())
    
    # Créer une colonne pour le type (chanson ou album)
    df['content_type'] = df.apply(
        lambda row: 'Album' if 'Album' in row['category'] 
        else 'Song' if any(word in row['category'] for word in ['Song', 'Record']) 
        else 'Other', 
        axis=1
    )
    
    # Créer une colonne pour le genre musical (si présent dans la catégorie)
    genres = ['Pop', 'Rock', 'R&B', 'Rap', 'Country', 'Jazz', 'Classical', 'Electronic', 'Dance', 'Latin', 'Gospel', 'Blues']
    df['genre'] = df['category'].apply(
        lambda x: next((genre for genre in genres if genre in x), 'Other')
    )
    
    # Créer une colonne pour la longueur du titre
    df['title_length'] = df['song_or_album'].apply(lambda x: len(x))
    
    # Créer une colonne pour le nombre de mots dans le titre
    df['title_word_count'] = df['song_or_album'].apply(lambda x: len(x.split()))
    
    # Utilisation de assign pour créer de nouvelles colonnes (pour répondre aux critères)
    df = df.assign(
        is_winner=df['winner'],
        nomination_decade=df['decade'],
        title_first_word=df['song_or_album'].apply(lambda x: x.split()[0] if x and len(x.split()) > 0 else ""),
        artist_name_length=df['main_artist'].apply(len)
    )
    
    # Calculer le taux de victoire par artiste (pour utiliser sum)
    artist_wins = df[df['winner'] == True].groupby('main_artist').size()
    artist_nominations = df.groupby('main_artist').size()
    artist_win_rates = (artist_wins / artist_nominations * 100).fillna(0)
    
    # Ajouter le taux de victoire comme nouvelle colonne
    df['artist_win_rate'] = df['main_artist'].map(artist_win_rates)
    
    return df

# Fonction pour extraire les mots les plus fréquents
def extract_common_words(text_series, n=10):
    # Télécharger les stopwords si nécessaire
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    additional_stops = {'feat', 'ft', 'the', 'and', 'a', 'an', 'in', 'on', 'of', 'to', 'for', 'with', 'by'}
    stop_words.update(additional_stops)
    
    words = []
    for text in text_series.dropna():
        if isinstance(text, str):
            # Nettoyer le texte et extraire les mots
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            words.extend([word for word in clean_text.split() if word not in stop_words and len(word) > 2])
    
    return Counter(words).most_common(n)

# Fonction pour analyser les données avec l'API Mistral
def analyze_with_mistral(df, analysis_type, api_key):
    """
    Fonction qui utilise l'API Mistral pour analyser les données Grammy
    """
    try:
        # Préparer les données pour l'analyse
        # Limiter la taille des données pour éviter de dépasser les limites de l'API
        data_summary = {
            "total_entries": len(df),
            "unique_artists": df['main_artist'].nunique(),
            "unique_categories": df['category'].nunique(),
            "top_artists": df.groupby('main_artist').size().sort_values(ascending=False).head(5).to_dict(),
            "top_winners": df[df['winner'] == True].groupby('main_artist').size().sort_values(ascending=False).head(5).to_dict(),
            "decade_distribution": df.groupby('decade').size().to_dict(),
            "genre_distribution": df.groupby('genre').size().to_dict(),
            "category_distribution": df.groupby('category').size().sort_values(ascending=False).head(10).to_dict()
        }
        
        # Créer des prompts spécifiques selon le type d'analyse
        if analysis_type == "tendances_genres":
            prompt = f"""
            Analyse les tendances des genres musicaux dans les Grammy Awards à partir des données suivantes:
            
            Distribution par genre: {data_summary['genre_distribution']}
            Distribution par décennie: {data_summary['decade_distribution']}
            
            Fais une analyse détaillée des tendances des genres musicaux au fil du temps.
            Identifie les genres dominants par décennie et les tendances émergentes.
            Présente ton analyse sous forme de points clés avec des titres et sous-titres.
            Utilise un format markdown avec des listes à puces et des mises en évidence.
            """
        
        elif analysis_type == "succes_artistes":
            # Calculer quelques statistiques supplémentaires pour enrichir l'analyse
            win_rates = {}
            for artist in data_summary['top_artists']:
                artist_data = df[df['main_artist'] == artist]
                nominations = len(artist_data)
                wins = len(artist_data[artist_data['winner'] == True])
                if nominations > 0:
                    win_rates[artist] = (wins / nominations) * 100
            
            prompt = f"""
            Analyse les facteurs de succès des artistes aux Grammy Awards à partir des données suivantes:
            
            Top artistes (nombre de nominations): {data_summary['top_artists']}
            Top gagnants: {data_summary['top_winners']}
            Taux de victoire des artistes populaires: {win_rates}
            
            Fais une analyse détaillée des facteurs qui contribuent au succès des artistes aux Grammy Awards.
            Identifie les artistes les plus couronnés de succès et leurs caractéristiques communes.
            Présente ton analyse sous forme de points clés avec des titres et sous-titres.
            Utilise un format markdown avec des listes à puces et des mises en évidence.
            """
        
        elif analysis_type == "evolution_categories":
            prompt = f"""
            Analyse l'évolution des catégories des Grammy Awards à partir des données suivantes:
            
            Distribution des catégories: {data_summary['category_distribution']}
            Distribution par décennie: {data_summary['decade_distribution']}
            Nombre total de catégories: {data_summary['unique_categories']}
            
            Fais une analyse détaillée de l'évolution des catégories des Grammy Awards au fil du temps.
            Identifie comment les catégories ont évolué et ce que cela révèle sur l'industrie musicale.
            Présente ton analyse sous forme de points clés avec des titres et sous-titres.
            Utilise un format markdown avec des listes à puces et des mises en évidence.
            """
        
        else:  # Analyse générale
            prompt = f"""
            Analyse générale des données des Grammy Awards:
            
            Statistiques générales:
            - Total des entrées: {data_summary['total_entries']}
            - Nombre d'artistes uniques: {data_summary['unique_artists']}
            - Nombre de catégories: {data_summary['unique_categories']}
            
            Top artistes: {data_summary['top_artists']}
            Top gagnants: {data_summary['top_winners']}
            Distribution par genre: {data_summary['genre_distribution']}
            Distribution par décennie: {data_summary['decade_distribution']}
            
            Fais une analyse générale des Grammy Awards à partir de ces données.
            Identifie les tendances principales, les faits marquants et les insights intéressants.
            Présente ton analyse sous forme de points clés avec des titres et sous-titres.
            Utilise un format markdown avec des listes à puces et des mises en évidence.
            """
        
        # Appel à l'API Mistral
        url = "https://api.mistral.ai/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": "mistral-small",  # Utiliser le modèle Mistral Small qui est plus rapide et moins coûteux
            "messages": [
                {"role": "system", "content": "Tu es un analyste de données spécialisé dans l'industrie musicale et les Grammy Awards. Tu fournis des analyses claires, concises et pertinentes basées sur les données fournies."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            return analysis
        else:
            return f"Erreur lors de l'appel à l'API Mistral: {response.status_code}\n{response.text}"
    
    except Exception as e:
        return f"Erreur lors de l'analyse avec Mistral AI: {str(e)}"

# Fonction principale
def main():
    # Charger les données
    df = load_data()
    
    if df is None:
        st.error("Impossible de charger les données. Veuillez vérifier que le fichier 'grammy_winners.csv' est présent.")
        return
    
    # Prétraiter les données
    df = preprocess_data(df)
    
    # En-tête de l'application
    st.markdown('<h1 class="main-header">🏆 Grammy Awards Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Une analyse interactive des Grammy Awards à travers les années</p>', unsafe_allow_html=True)
    
    # Sidebar pour les filtres
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/7/79/Grammy_Award_logo.svg/1200px-Grammy_Award_logo.svg.png", width=200)
    st.sidebar.title("Filtres")
    
    # Filtres
    year_range = st.sidebar.slider(
        "Années", 
        min_value=int(df['year'].min()), 
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max()))
    )
    
    # Pour les catégories:
    category_options = [str(cat) for cat in df['category'].unique() if pd.notna(cat)]
    selected_categories = st.sidebar.multiselect(
        "Catégories",
        options=sorted(category_options),
        default=[]
    )
    
    # Pour les artistes:
    artist_options = [str(artist) for artist in df['main_artist'].unique() if pd.notna(artist)]
    selected_artists = st.sidebar.multiselect(
        "Artistes",
        options=sorted(artist_options),
        default=[]
    )
    
    winner_filter = st.sidebar.radio(
        "Statut",
        options=["Tous", "Gagnants", "Nominés"]
    )
    
    # Ajout de l'option pour exclure Unknown et Other
    st.sidebar.markdown("---")
    st.sidebar.subheader("Options d'affichage")
    exclude_unknown = st.sidebar.checkbox("Exclure 'Unknown' et 'Other' des analyses", value=False)
    
    # Appliquer les filtres
    filtered_df = df.copy()
    
    filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    if selected_artists:
        filtered_df = filtered_df[filtered_df['main_artist'].isin(selected_artists)]
    
    if winner_filter == "Gagnants":
        filtered_df = filtered_df[filtered_df['winner'] == True]
    elif winner_filter == "Nominés":
        filtered_df = filtered_df[filtered_df['winner'] == False]
    
    # Appliquer le filtre pour exclure Unknown et Other si demandé
    if exclude_unknown:
        filtered_df = filtered_df[
            ~((filtered_df['main_artist'] == 'Unknown') | 
              (filtered_df['main_artist'] == 'Other') |
              (filtered_df['category'] == 'Other') |
              (filtered_df['genre'] == 'Other'))
        ]
    
    # Afficher les métriques clés
    st.markdown('<h2 class="sub-header">Métriques clés</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Nombre total d'entrées", len(filtered_df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Nombre de gagnants", len(filtered_df[filtered_df['winner'] == True]))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Nombre d'artistes uniques", filtered_df['main_artist'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Nombre de catégories", filtered_df['category'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Créer des onglets pour différentes analyses
    tabs = st.tabs(["Vue d'ensemble", "Analyse des artistes", "Analyse des catégories", "Analyse IA", "Données brutes"])
    
    # Onglet Vue d'ensemble
    with tabs[0]:
        st.markdown('<h2 class="sub-header">Vue d\'ensemble des Grammy Awards</h2>', unsafe_allow_html=True)
        
        # Graphique 1: Évolution du nombre de nominations par année
        st.subheader("Évolution du nombre de nominations par année")
        nominations_by_year = filtered_df.groupby('year').size()
        st.bar_chart(nominations_by_year)
        st.caption("Ce graphique montre l'évolution du nombre total de nominations par année.")
        
        # Utilisation de sum pour calculer le nombre total de gagnants par décennie
        winners_by_decade = filtered_df.groupby('decade')['is_winner'].sum()
        
        # Graphique 2: Distribution des gagnants par décennie
        st.subheader("Distribution des gagnants par décennie")
        st.bar_chart(winners_by_decade)
        st.caption("Ce graphique montre la distribution des gagnants par décennie.")
        
        # Graphique 3: Distribution par genre musical
        st.subheader("Distribution par genre musical")
        genre_counts = filtered_df.groupby('genre').size()
        st.bar_chart(genre_counts)
        st.caption("Ce graphique montre la répartition des nominations par genre musical.")
        
        # Utilisation de value_counts pour la distribution par type de contenu
        content_type_counts = filtered_df['content_type'].value_counts()
        
        # Graphique 4: Distribution par type de contenu
        st.subheader("Distribution par type de contenu")
        st.bar_chart(content_type_counts)
        st.caption("Ce graphique montre la répartition des nominations entre albums, chansons et autres catégories.")
        
        # Insights
        st.markdown('<div class="insight-text">', unsafe_allow_html=True)
        st.markdown("### Insights clés")
        
        # Calculer quelques insights
        most_awarded_year = filtered_df[filtered_df['winner'] == True].groupby('year').size().sort_values(ascending=False).index[0]
        most_competitive_year = filtered_df.groupby('year').size().sort_values(ascending=False).index[0]
        
        st.markdown(f"- L'année avec le plus de récompenses est **{most_awarded_year}**.")
        st.markdown(f"- L'année la plus compétitive (avec le plus de nominations) est **{most_competitive_year}**.")
        
        # Tendance récente
        recent_years = filtered_df[filtered_df['year'] >= 2010]
        if len(recent_years) > 0:
            top_recent_genre = recent_years.groupby('genre').size().sort_values(ascending=False).index[0]
            st.markdown(f"- Le genre musical le plus représenté récemment est **{top_recent_genre}**.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Onglet Analyse des artistes
    with tabs[1]:
        st.markdown('<h2 class="sub-header">Analyse des artistes</h2>', unsafe_allow_html=True)
        
        # Top artistes par nombre de nominations
        st.subheader("Top 10 des artistes par nombre de nominations")
        top_artists_nominations = filtered_df.groupby('main_artist').size().sort_values(ascending=False).head(10)
        st.bar_chart(top_artists_nominations)
        st.caption("Ce graphique montre les 10 artistes ayant reçu le plus de nominations.")
        
        # Top artistes par nombre de victoires
        st.subheader("Top 10 des artistes par nombre de victoires")
        top_artists_wins = filtered_df[filtered_df['winner'] == True].groupby('main_artist').size().sort_values(ascending=False).head(10)
        st.bar_chart(top_artists_wins)
        st.caption("Ce graphique montre les 10 artistes ayant remporté le plus de Grammy Awards.")
        
        # Taux de victoire par artiste (utilisation de map)
        st.subheader("Taux de victoire des artistes les plus nominés")
        
        # Calculer le taux de victoire pour les artistes avec au moins 5 nominations
        artist_stats = filtered_df.groupby('main_artist').agg({
            'winner': ['count', 'sum']
        })
        artist_stats.columns = ['nominations', 'wins']
        artist_stats = artist_stats[artist_stats['nominations'] >= 5]
        artist_stats['win_rate'] = (artist_stats['wins'] / artist_stats['nominations'] * 100).round(1)
        artist_stats = artist_stats.sort_values('win_rate', ascending=False).head(15)
        
        st.bar_chart(artist_stats['win_rate'])
        st.caption("Ce graphique montre le taux de victoire des artistes les plus nominés (minimum 5 nominations).")
        
        # Tableau des artistes
        st.subheader("Tableau des artistes les plus nominés")
        
        # Créer un DataFrame pour le tableau
        top_artists_df = pd.DataFrame({
            'Artiste': top_artists_nominations.index,
            'Nominations': top_artists_nominations.values
        })
        
        # Ajouter les victoires si disponibles
        if len(top_artists_wins) > 0:
            wins_dict = dict(zip(top_artists_wins.index, top_artists_wins.values))
            top_artists_df['Victoires'] = top_artists_df['Artiste'].map(wins_dict).fillna(0).astype(int)
            top_artists_df['Taux de victoire (%)'] = (top_artists_df['Victoires'] / top_artists_df['Nominations'] * 100).round(1)
        
        st.dataframe(top_artists_df, use_container_width=True)
    
    # Onglet Analyse des catégories
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Analyse des catégories</h2>', unsafe_allow_html=True)
        
        # Top catégories par nombre de nominations
        st.subheader("Top 15 des catégories par nombre de nominations")
        top_categories = filtered_df.groupby('category').size().sort_values(ascending=False).head(15)
        st.bar_chart(top_categories)
        st.caption("Ce graphique montre les 15 catégories ayant reçu le plus de nominations.")
        
        # Analyse des relations entre catégories et artistes
        st.subheader("Top relations entre artistes et catégories")
        
        # Créer un tableau des relations les plus fortes
        if len(filtered_df) > 0:
            # Compter les occurrences de chaque paire artiste-catégorie
            artist_category_counts = filtered_df.groupby(['main_artist', 'category']).size().reset_index(name='Nombre de nominations')
            artist_category_counts = artist_category_counts.sort_values('Nombre de nominations', ascending=False).head(20)
            
            # Renommer les colonnes pour l'affichage
            artist_category_counts.columns = ['Artiste', 'Catégorie', 'Nombre de nominations']
            
            st.dataframe(artist_category_counts, use_container_width=True)
            st.caption("Ce tableau montre les 20 relations les plus fortes entre artistes et catégories (nombre de nominations).")
    
    # Onglet Analyse IA
    with tabs[3]:
        st.markdown('<h2 class="sub-header">Analyse avec Intelligence Artificielle (Mistral AI)</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Cette section utilise l'intelligence artificielle de Mistral AI pour analyser les données des Grammy Awards et 
        fournir des insights avancés basés sur les filtres que vous avez appliqués.
        """)
        
        # Options d'analyse
        analysis_options = {
            "analyse_generale": "Analyse générale des données",
            "tendances_genres": "Tendances des genres musicaux",
            "succes_artistes": "Facteurs de succès des artistes",
            "evolution_categories": "Évolution des catégories"
        }
        
        selected_analysis = st.selectbox(
            "Choisissez un type d'analyse",
            options=list(analysis_options.keys()),
            format_func=lambda x: analysis_options[x]
        )
        
        # Clé API Mistral (pré-remplie avec la clé fournie)
        api_key = st.text_input("Clé API Mistral", value="WfLdVtMjtWdMW0AYMzz4XrFsoVvm6t56", type="password")
        
        # Bouton pour lancer l'analyse
        if st.button("Analyser avec Mistral AI"):
            if api_key:
                with st.spinner("Mistral AI analyse les données..."):
                    # Appeler la fonction d'analyse avec l'API Mistral
                    ai_response = analyze_with_mistral(filtered_df, selected_analysis, api_key)
                    
                    # Afficher les résultats
                    st.markdown('<div class="insight-text">', unsafe_allow_html=True)
                    st.markdown(ai_response)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Veuillez entrer une clé API Mistral valide pour continuer.")
    
    # Onglet Données brutes
    with tabs[4]:
        st.markdown('<h2 class="sub-header">Données brutes</h2>', unsafe_allow_html=True)
        
        # Afficher les données filtrées
        st.dataframe(filtered_df, use_container_width=True)
        
        # Option pour télécharger les données
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les données filtrées (CSV)",
            data=csv,
            file_name=f"grammy_awards_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Pied de page
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("Développé pour le projet d'analyse de données | Grammy Awards Explorer | 2024")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()