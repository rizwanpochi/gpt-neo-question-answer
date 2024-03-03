import pandas as pd
import os
from transformers import pipeline
from model_loader import generate_text 


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(ROOT_DIR, 'data/Financebench.csv')
# Load the CSV data into a DataFrame
df = pd.read_csv(file_path)


def find_relevant_data(question):

    relevant_rows = df[df['question'].str.contains(question, case=False, na=False)]
    if not relevant_rows.empty:
        return relevant_rows.iloc[0]['answer']  # Return the first match
    return None



def get_answer(question):
    # Attempt to find an answer from the CSV data
    answer_from_csv = find_relevant_data(question)
    if answer_from_csv is not None:

        return answer_from_csv
    else:

        return generate_text(question)


