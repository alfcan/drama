import pandas as pd
from mutation.mutation_operator import MutationOperator

if __name__ == '__main__':
    data = {
        'Age': [25, 30, 45, 50, 23],
        'Income': [50000, 60000, 80000, 90000, 45000],
        'Description': ["An engineer designs, builds, and maintains complex systems and structures. They apply principles of mathematics and science to create efficient solutions to technical problems.", "A doctor diagnoses and treats illnesses, injuries, and other medical conditions. They provide essential healthcare services, conduct medical examinations, and prescribe treatments.", "An artist creates visual or performing art pieces, such as paintings, sculptures, music, or dance. They express emotions, ideas, and concepts through their creative work."
, "A lawyer provides legal advice, represents clients in court, and prepares legal documents. They specialize in areas like criminal law, corporate law, or family law."
, "A nurse provides essential care to patients, assists doctors, and ensures the smooth operation of healthcare facilities. They monitor patient health, administer medications, and offer support."],
        
    }
    df = pd.DataFrame(data)

    mutation_operator = MutationOperator(df.copy())

    numeric_operators = {
        # 'increment_decrement_feature': lambda df, col: mutation_operator.increment_decrement_feature(col, increment=True, amount=5, percentage=50),
        # 'swap_values': lambda df, col: mutation_operator.swap_values(col, num_swaps=2),
        # 'scale_values': lambda df, col: mutation_operator.scale_values(col, scale_factor=1.1, percentage=50),
        # 'discrete_binning': lambda df, col: mutation_operator.discrete_binning(col, bins=3)
    }

    text_operators = {
        # 'random_category_assignment': lambda df, col: mutation_operator.random_category_assignment(col, percentage=50),
        'augment_text': lambda df, col: mutation_operator.augment_text(col, percentage=40),
        # 'replace_synonyms': lambda df, col: mutation_operator.replace_synonyms(col, row_percentage=40, word_percentage=40),
        # 'add_noise': lambda df, col: mutation_operator.add_noise(col, noise_chance=0.2),
        # 'swap_values': lambda df, col: mutation_operator.swap_values(col, num_swaps=2)
    }

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            operators = numeric_operators
        else:
            operators = text_operators

        for op_name, op_func in operators.items():
            df_before = df.copy()
            df_after = op_func(df.copy(), col)
            print(f"\n\nApplying {op_name} on column '{col}':")
            print("\nBefore mutation:")
            print(df_before)
            print("\nAfter mutation:")
            print(df_after)