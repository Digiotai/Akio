import pandas as pd
from vyzeai.models.openai import ChatOpenAI


def generate_synthetic_data(api_key, file_path, num_rows=10, chunk_size=50):
    """Generate synthetic data."""

    llm = ChatOpenAI(api_key)

    data = pd.read_excel(file_path).tail(50)
    sample_str = data.to_csv(index=False, header=False)

    sysp = "You are a synthetic data generator. Your output should only be specified format without any additional text and code fences."

    generated_rows = []
    rows_generated = 0

    while rows_generated < num_rows:

        if generated_rows:
            current_sample_str = "\n".join([",".join(row) for row in generated_rows[-50:]])
        else:
            current_sample_str = sample_str

        rows_to_generate = min(chunk_size, num_rows - rows_generated)

        prompt = (
            f"Generate {rows_to_generate} rows of synthetic data based on the structure and distribution of the following sample:\n\n{current_sample_str}\n"
            "\nEnsure the new rows are realistic, varied, and maintain the same data types, distribution, and logical relationships. "
            "Format as pipe-separated values ('|') without including column names or old data.")

        generated_data = llm.run(prompt, system_message=sysp)

        rows = [row.split("|") for row in generated_data.strip().split("\n") if row]

        rows_needed = num_rows - rows_generated
        generated_rows.extend(rows[:rows_needed])

        rows_generated += len(rows[:rows_needed])

    generated_df = pd.DataFrame(generated_rows, columns=data.columns)

    return generated_df


def generate_data_from_text(api_key, text_sample, column_names, num_rows=10, chunk_size=50):
    llm = ChatOpenAI(api_key)

    sysp = "You are a data generator that produces only specified formatted data with no extra text or code fences."

    generated_rows = []
    rows_generated = 0

    column_names_str = ", ".join(column_names)

    while rows_generated < num_rows:
        rows_to_generate = min(chunk_size, num_rows - rows_generated)

        if rows_generated == 0:
            prompt = (f"Based on the following description:\n'{text_sample}'\n"
                      f"Generate {rows_to_generate} rows of synthetic data with the following columns:\n"
                      f"Columns: {column_names_str}\n"
                      "Ensure that all columns are present and the data is realistic, varied, and maintains logical relationships. "
                      "Format the data as tilde-separated values ('~') without including column names or any extra text.")
        else:
            reference_data = "\n".join([",".join(row) for row in generated_rows[-5:]])

            prompt = (f"Based on the following description:\n'{text_sample}'\n"
                      f"Generate {rows_to_generate} rows of synthetic data with the following columns:\n"
                      f"Columns: {column_names_str}\n"
                      f"Follow the format of these recently generated rows:\n{reference_data}\n"
                      "Ensure taht all columns are present and the data is realistic, varied, and maintains logical relationships. "
                      "Format the data as tilde-separated values ('~') without including column names or any extra text.")

        generated_data = llm.run(prompt, system_message=sysp)
        print(generated_data)

        rows = [row.split("~") for row in generated_data.strip().split("\n") if row]
        print(rows[:5])

        rows_needed = num_rows - rows_generated
        generated_rows.extend(rows[:rows_needed])

        rows_generated += len(rows[:rows_needed])

    df = pd.DataFrame(generated_rows, columns=column_names)

    return df


def fill_missing_data_in_chunk(api_key, file_path, chunk_size=50):
    llm = ChatOpenAI(api_key)

    # Load the data
    data = pd.read_excel(file_path)

    # Prepare a chunk of the data (last 'chunk_size' rows)
    data_chunk = data.tail(chunk_size).fillna('null')

    # Convert the chunk to CSV format (without headers)
    sample_str = data_chunk.to_csv(index=False, header=False)

    sysp = "You are a data completion assistant. Your output should only be specified format without any additional text and code fences."

    # Create a prompt asking to fill any missing values in the chunk of data
    prompt = (f"Here is a dataset with some missing values:\n\n{sample_str}\n"
              "Please fill in the missing values based on the distribution of the existing data. Ensure the new values are realistic, "
              "maintain the same data types, and align with the logical relationships in the dataset. "
              "Format the output as pipe-separated values ('|') without including column names.")

    # Generate filled data from the model
    generated_data = llm.run(prompt, system_message=sysp)

    # Parse the generated data back into a DataFrame
    filled_chunk = pd.DataFrame(
        [row.split("|") for row in generated_data.strip().split("\n")],
        columns=data.columns
    )

    # Replace the last 'chunk_size' rows in the original DataFrame with the filled chunk
    data.iloc[-chunk_size:] = filled_chunk

    return data

