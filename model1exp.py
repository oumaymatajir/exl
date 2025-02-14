import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PowerTransformer
import matplotlib.pyplot as plt

# Charger le modèle entraîné
model_filename = "modele_gradient_boosting_lasso.pkl"
models = joblib.load(model_filename)

# Initialiser les transformateurs
scaler = StandardScaler()
power_transformer = PowerTransformer()

# Interface utilisateur Streamlit
st.title("Prédiction des compositions de polymères recyclés")
st.write("Entrez les valeurs des caractéristiques pour obtenir les prédictions des proportions de PP, Fibre de verre, Peroxyde et Autres charges.")

# Entrées utilisateur
melt = st.number_input("Melt (g/10min)", min_value=0.0, format="%.2f")
izod = st.number_input("Izod (ft-lb/in)", min_value=0.0, format="%.2f")
traction = st.number_input("Traction (N)", min_value=0.0, format="%.2f")
flexion = st.number_input("Flexion (psi)", min_value=0.0, format="%.2f")

# Bouton de prédiction
if st.button("Prédire"):
    # Préparation des données d'entrée
    input_data = np.array([[melt, izod, traction, flexion]])
    input_scaled = scaler.fit_transform(input_data)
    input_transformed = power_transformer.fit_transform(input_scaled)
    
    # Prédictions
    predictions = {}
    for col in models.keys():
        predictions[col] = models[col].predict(input_transformed)[0]
    
    # Affichage des résultats
    st.subheader("Résultats de la prédiction :")
    results_df = pd.DataFrame(predictions, index=["Valeur prédite (%)"])
    st.dataframe(results_df.style.format("{:.4f}"))
    
    # Affichage en graphique à barres
    st.subheader("Visualisation des résultats")
    fig, ax = plt.subplots()
    ax.bar(predictions.keys(), predictions.values(), color=['blue', 'green', 'red', 'purple'])
    ax.set_ylabel("Pourcentage estimé (%)")
    ax.set_title("Distribution des proportions prédictes")
    st.pyplot(fig)
    
    # Option d'exportation des résultats
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Télécharger les résultats en CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# Lancer Streamlit via la ligne de commande :
# streamlit run app.py
