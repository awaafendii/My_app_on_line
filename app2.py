import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Détection de Fraude Interactive", page_icon="🕵️‍♂️", layout="wide")

# --- 2. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
@st.cache_data
def load_data():
    # Chargement du dataset (issu de votre premier code)
    df = pd.read_csv("Dataset.csv")
    
    # Nettoyage et ingénierie des caractéristiques
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Date'] = df['TransactionStartTime'].dt.date # Ajout pour le filtre temporel
    df['Heure'] = df['TransactionStartTime'].dt.hour
    df['Jour'] = df['TransactionStartTime'].dt.day_name()
    df['Mois'] = df['TransactionStartTime'].dt.month
    df['Annee'] = df['TransactionStartTime'].dt.year
    df['Valeur_absolue'] = df['Amount'].abs()
    
    # Mapping explicite de la cible
    if 'FraudResult' in df.columns:
        df['Fraude'] = df['FraudResult'].map({0: 'Légitime', 1: 'Frauduleuse'})
    
    # Ordre logique pour les jours de la semaine
    jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Jour'] = pd.Categorical(df['Jour'], categories=jours_ordre, ordered=True)
    
    return df

df = load_data()

st.title("Dashboard Interactif - Analyse et Détection de Fraudes")

# --- 3. BARRE LATÉRALE - FILTRES DYNAMIQUES ---
st.sidebar.header("Filtres de données")

# Filtre par Date
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    date_min = df['Date'].min().date()
    date_max = df['Date'].max().date()
    date_range = st.sidebar.date_input("Filtrer par Date", [date_min, date_max])

    if len(date_range) == 2:
        df = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]

# Filtres catégoriels multiples (auto-détectés)
colonnes_categorique = df.select_dtypes(include=['object', 'category']).columns.tolist()
# On retire 'Date' s'il a été listé par erreur comme objet
if 'Date' in colonnes_categorique: colonnes_categorique.remove('Date')

for col in colonnes_categorique:
    valeurs = df[col].dropna().unique().tolist()
    selection = st.sidebar.multiselect(f"Filtrer {col}", valeurs, default=valeurs)
    df = df[df[col].isin(selection)]


# --- 4. BARRE LATÉRALE - PARAMÈTRES DES GRAPHIQUES ---
st.sidebar.markdown("---")
st.sidebar.header("Paramètres des Graphiques")
colonnes_numerique = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()

# Sélecteurs dynamiques
col_x = st.sidebar.selectbox("Variable X (catégorique)", colonnes_categorique, index=colonnes_categorique.index('Fraude') if 'Fraude' in colonnes_categorique else 0)
col_y = st.sidebar.selectbox("Variable Y (numérique)", colonnes_numerique, index=colonnes_numerique.index('Valeur_absolue') if 'Valeur_absolue' in colonnes_numerique else 0)
col_color = st.sidebar.selectbox("Variable couleur (optionnel)", [None] + colonnes_categorique, index=colonnes_categorique.index('Fraude') + 1 if 'Fraude' in colonnes_categorique else 0)

# --- 5. INDICATEURS CLÉS (KPIs) ---
st.markdown("### 📊 Chiffres Clés (Données filtrées)")
col1, col2, col3, col4 = st.columns(4)

total_transactions = len(df)
total_volume = df['Valeur_absolue'].sum() if 'Valeur_absolue' in df.columns else 0
total_fraudes = len(df[df['Fraude'] == 'Frauduleuse']) if 'Fraude' in df.columns else 0
taux_fraude = (total_fraudes / total_transactions) * 100 if total_transactions > 0 else 0

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Volume Total (Absolu)", f"{total_volume:,.0f}")
col3.metric("Nombre de Fraudes", f"{total_fraudes}")
col4.metric("Taux de Fraude", f"{taux_fraude:.2f} %")

st.divider()

# --- 6. AFFICHAGE DES GRAPHIQUES DYNAMIQUES ---

# Ligne de tendance par date
if 'Date' in df.columns:
    st.subheader(f"Évolution temporelle de la moyenne de : {col_y}")
    line_data = df.groupby('Date')[col_y].mean().reset_index()
    fig_line = px.line(line_data, x='Date', y=col_y, markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

colA, colB = st.columns(2)

with colA:
    # Histogramme
    st.subheader(f"Distribution de : {col_y}")
    fig_hist = px.histogram(df, x=col_y, color=col_color, nbins=30, barmode='group')
    st.plotly_chart(fig_hist, use_container_width=True)

with colB:
    # Barplot (moyenne par catégorie)
    st.subheader(f"Moyenne de {col_y} par {col_x}")
    agg_data = df.groupby(col_x, observed=False)[col_y].mean().reset_index()
    fig_bar = px.bar(agg_data, x=col_x, y=col_y, color=col_x)
    st.plotly_chart(fig_bar, use_container_width=True)

colC, colD = st.columns(2)

with colC:
    # Heatmap de corrélation
    st.subheader("Matrice de Corrélation")
    # On évite les erreurs s'il y a moins de 2 variables numériques
    if len(colonnes_numerique) > 1:
        corr = df[colonnes_numerique].corr()
        fig_corr, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig_corr)
    else:
        st.info("Pas assez de variables numériques pour la corrélation.")

with colD:
    # Pie chart
    if col_x:
        st.subheader(f"Répartition globale par : {col_x}")
        fig_pie = px.pie(df, names=col_x, hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# --- 7. ANALYSE TABULAIRE ET EXPORT ---
st.subheader("Analyse Tabulaire Avancée")
groupby_col = st.selectbox("Grouper les données par :", colonnes_categorique)
agg_col = st.multiselect("Colonnes numériques à agréger :", colonnes_numerique, default=[col_y])

if groupby_col and agg_col:
    resume = df.groupby(groupby_col, observed=False)[agg_col].agg(['mean', 'sum', 'count']).round(2)
    st.dataframe(resume, use_container_width=True)

# Données brutes
with st.expander("Aperçu des données brutes filtrées"):
    st.dataframe(df)

# Téléchargement
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Télécharger les données filtrées au format CSV", 
    data=csv, 
    file_name="transactions_fraude_filtrees.csv", 
    mime="text/csv"
)