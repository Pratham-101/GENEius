import numpy as np
import pandas as pd

import disease_ml


def get_disease_prs(features):

    prs_values = {}

    features['risk_allele_frequency'] = ((features['allele1'] == features['risk_allele']).astype(float) + (features['allele2'] == features['risk_allele']).astype(float))

    refined_features = features.dropna(subset=['odds_ratio']).copy()
    refined_features['mrs'] = features['risk_allele_frequency'] * np.log(features['odds_ratio'])

    prs_values = refined_features.groupby('disease_name')['mrs'].sum().to_dict()
    return prs_values


def  get_disease_prediction(features, prs_values):

    final_features = pd.DataFrame()
    final_features['key'] = features['rsid']+ "_" + features['disease_name']

    final_features['risk_allele_frequency'] = features['risk_allele_frequency'] / 2

    final_features = final_features.transpose()
    final_features.columns = final_features.iloc[0]
    final_features = final_features.drop(final_features.index[0]).reset_index(drop=True)

    predictions = disease_ml.model_output(final_features)
    return predictions


def get_disease_score():
    pass