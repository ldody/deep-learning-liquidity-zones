import os, sys
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import random

cwd = os.getcwd()
files = os.listdir(cwd)

""" --- Loading datasets --- """
mail_df = pd.read_csv('Emails_Sat.csv', sep=';')
enquete_df = pd.read_csv('Rep_Enquetes.csv', sep=';')
achat_df = pd.read_csv('Achats_Retraitement.csv', sep=';')

achat_df = achat_df.loc[(achat_df['Total_Articles_Quantity'] != 0) & (achat_df['Is_Return_Flag'] == False)]

c_id = 'New_Client_Id_Client'


""" --- Converting dates to datetime format --- """
df_ls = [achat_df, mail_df, enquete_df]

for df in df_ls:
	for name in df.columns:
		if 'date' in name.lower():
			df[name] = pd.to_datetime(df[name])
			
			
""" --- Function to link satisfaction email --- """
def func_satis(chunk_achat):
	
	client_id = chunk_achat['New_Client_Id_Client'].iloc[0]
	
	delai = 30
	
	data = []
	
	chunk_mail = mail_df[mail_df['New_Client_Id_Client'] == client_id]
	
	for _, a_date in chunk_achat.iterrows():
		for _, s_date in chunk_mail.iterrows():
			diff_days = (s_date['sending_date_group'] - a_date['PURCHASE_DATE']).days
				
			if 1 <= diff_days <= delai:
				label = (-1/(delai-1)) * diff_days + (delai/(delai-1))
				data.append([client_id, a_date['PURCHASE_DATE'], s_date['sending_date_group'], diff_days, 1/diff_days, label])
	
	return pd.DataFrame(data, columns=['New_Client_Id_Client', 'PURCHASE_DATE', 'sending_date_group', 'diff_day', '1/diff_days', 'label'])

""" --- Function to link reminder email --- """
def func_rel(chunk_achat):
	
	client_id = chunk_achat['New_Client_Id_Client'].iloc[0]
	
	delai = 2
	
	data = []
	
	chunk_mail = mail_df[mail_df['New_Client_Id_Client'] == client_id]
	
	for _, s_date in chunk_achat.iterrows():
		for _, r_date in chunk_mail.iterrows():
			diff_days = (r_date['sending_date_group'] - s_date['sending_date_group']).days
			if 5 - delai <= diff_days <= 5 + delai:
				label = (diff_days * 0.5 - 1.5 if diff_days < 5 else - 0.5 * diff_days + 3.5)
				data.append([client_id, s_date['sending_date_group'], r_date['sending_date_group'], diff_days, 1/diff_days, label])

	
	return pd.DataFrame(data, columns=['New_Client_Id_Client', 'sending_date_group', 'rel_date_group', 'diff_day_rel', '1/diff_days_rel', 'label_rel'])
	

""" --- Linking mails to purchase orders --- """
results = Parallel(n_jobs=8)(delayed(func_satis)(chunk_achat) for _, chunk_achat in achat_df.groupby(c_id, as_index=False))
prob = pd.concat(results, ignore_index=True)

results = Parallel(n_jobs=8)(delayed(func_rel)(chunk_achat) for _, chunk_achat in prob.groupby(c_id, as_index=False))
prob2 = pd.concat(results, ignore_index=True)

""" --- Merging satisfaction and reminder emails --- """
merged = pd.merge(prob, prob2, how='left', on=['New_Client_Id_Client','sending_date_group'])
merged = pd.concat([merged, prob]).drop_duplicates()
merged['confidence'] = merged[['label','label_rel']].mean(axis=1)

""" --- GA algorithm --- """
final_res = pd.DataFrame()

# Fonction de fitness
def evaluer_solution(solution, df):
	# Sélectionner les lignes correspondant aux 1 dans la solution
	try:
		lignes_selectionnees = df.iloc[np.where(solution == 1)[0]]
	except:
		return 0
	
	valeurs_uniques_purchase_date = df['PURCHASE_DATE'].unique()
	if not all(valeur in lignes_selectionnees['PURCHASE_DATE'].values for valeur in valeurs_uniques_purchase_date):
		return 0  # Si une valeur unique de PURCHASE_DATE est manquante
	
	# 1. Vérifier les doublons de dates dans les colonnes Date1 et Date2
	dates = pd.concat([lignes_selectionnees['sending_date_group'], lignes_selectionnees['rel_date_group']])
	if len(dates) != len(dates.unique()):  # Si des doublons existent
		return 0  # Pénalisation maximale (solution invalide)
	
	# 2. Calculer le nombre de dates distinctes
	nb_dates_distinctes = len(dates.unique())
	
	# 3. Calculer la moyenne des probabilités
	moyenne_probabilite = lignes_selectionnees['confidence'].mean() if len(lignes_selectionnees) > 0 else 0
	
	# Fonction de fitness : maximiser les dates distinctes et la moyenne de probabilité
	fitness = nb_dates_distinctes + moyenne_probabilite
	return fitness

# Fonction pour générer une solution aléatoire
def generer_solution(taille_population, df):
	# Solution binaire avec le nombre exact de "1" correspondant aux valeurs uniques dans Col4
	solution = np.zeros(len(df['PURCHASE_DATE']))
	
	# Sélectionner une ligne pour chaque valeur unique de Col4
	for valeur in df['PURCHASE_DATE'].unique():
		# Sélectionner un index correspondant à une ligne ayant cette valeur unique dans Col4

		indices_choisis = df[df['PURCHASE_DATE'] == valeur].index
		solution[random.choice(indices_choisis)] = 1
		
	return solution

# Croisement
def croisement(sol1, sol2):
	point = random.randint(1, len(sol1) - 1)
	return np.concatenate((sol1[:point], sol2[point:]))

# Mutation
def mutation(solution):
	i = random.randint(0, len(solution) - 1)
	solution[i] = 1 - solution[i]  # Inverser la valeur de la position i
	return solution

# Algorithme génétique
def algorithme_genetique(df, taille_population=10, generations=50):
	taille_solution = len(df)  # Nombre de lignes dans le DataFrame
	population = [generer_solution(taille_population, df) for _ in range(taille_population)]
	
	for _ in range(generations):
		# Évaluation des solutions
		population = sorted(population, key=lambda sol: evaluer_solution(sol, df), reverse=True)
		
		# Nouvelle population (garder les meilleures solutions)
		nouvelle_population = population[:2]  # Garder les deux meilleures solutions
		
		while len(nouvelle_population) < taille_population:
			parent1, parent2 = random.sample(population[:5], 2)  # Sélectionner parmi les meilleures
			enfant = mutation(croisement(parent1, parent2))
			nouvelle_population.append(enfant)
		
		population = nouvelle_population
	
	# Retourner la meilleure solution trouvée
	meilleure_solution = max(population, key=lambda sol: evaluer_solution(sol, df))
	return meilleure_solution

# Exécution de l'algorithme génétique
def GA(df):
	df = df.reset_index(drop=True)
	if len(df) == 1:
		res = df
		
	else:    
		meilleure_solution = algorithme_genetique(df, taille_population=30, generations=60)
		#display("Meilleure solution trouvée :", meilleure_solution)

		# Affichage des lignes sélectionnées
		lignes_selectionnees = df.iloc[np.where(meilleure_solution == 1)[0]]
		#display("Lignes sélectionnées :\n", lignes_selectionnees)
		
		res = lignes_selectionnees
		
	if len(df['PURCHASE_DATE'].unique()) != len(res['PURCHASE_DATE'].unique()):
		print('error', df['New_Client_Id_Client'].iloc[0])
		
	return res

""" --- Executing GA --- """
results = Parallel(n_jobs=8)(delayed(GA)(df) for _, df in merged.groupby('New_Client_Id_Client'))
final_res = pd.concat(results, ignore_index=True)

""" --- Function to link survey --- """
def func_enquete(chunk_achat):

	client_id = chunk_achat['New_Client_Id_Client'].iloc[0]
	dates = pd.DataFrame()

	tmp_fin = final_res.loc[final_res['New_Client_Id_Client'] == client_id,['sending_date_group','rel_date_group']]
	tmp_fin['date'] = tmp_fin.apply(lambda row: row['rel_date_group'] if not pd.isna(row['rel_date_group']) else row['sending_date_group'], axis=1)
	
	dates = tmp_fin['date'].to_list()
	
	chunk_mail = mail_df.loc[(mail_df['New_Client_Id_Client'] == client_id) & (mail_df['is_clicked'] == True)]
	chunk_enquete = enquete_df[enquete_df['New_Client_Id_Client'] == client_id]

	tmp = chunk_mail.copy()[['New_Client_Id_Client','sending_date_group']]
	tmp['answer_date'] = tmp['sending_date_group'].apply(lambda x: chunk_enquete.loc[chunk_enquete['answer_date'] >= x,'answer_date'].tolist())
	tmp = tmp[tmp['sending_date_group'].isin(dates)].explode('answer_date')
	
	used_dates_ls = []
	
	def select_date(col):
		col = col[~col['answer_date'].isin(used_dates_ls)]
		date = col['answer_date'].min()
		used_dates_ls.append(date)
		
		return date
		
	
	tmp = tmp.groupby(['New_Client_Id_Client','sending_date_group'], as_index=False)[['answer_date']].apply(lambda col: select_date(col))
	tmp = tmp.dropna()
	tmp.columns = ['_'.join(col) for col in tmp.columns]
	
	if len(tmp) == 0:
		return pd.DataFrame()

	return tmp


""" --- Linking mails to surveys --- """
results = Parallel(n_jobs=8)(delayed(func_enquete)(chunk_achat) for _, chunk_achat in achat_df.groupby(c_id, as_index=False))
matching_mail_enq = pd.concat(results, ignore_index=True)

""" --- Final matching --- """
matching_mail_enq = matching_mail_enq.rename(columns={'sending_date_group': 'date'})

rel = final_res[~final_res['rel_date_group'].isna()]
satis = final_res[final_res['rel_date_group'].isna()]

merged_rel = pd.merge(rel, 
					  matching_mail_enq, 
					  how='left',
					  left_on=['New_Client_Id_Client','rel_date_group'], 
					  right_on=['New_Client_Id_Client','date'])

merged_satis = pd.merge(satis,
						matching_mail_enq, 
						how='left', 
						left_on=['New_Client_Id_Client','sending_date_group'], 
						right_on=['New_Client_Id_Client','date'])

final = pd.concat([merged_rel, merged_satis], ignore_index=True).sort_values(['New_Client_Id_Client','PURCHASE_DATE'])

final[['New_Client_Id_Client','PURCHASE_DATE','sending_date_group','rel_date_group','answer_date']].to_csv('res.csv')