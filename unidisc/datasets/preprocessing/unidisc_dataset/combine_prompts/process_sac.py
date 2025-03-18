import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('sac_public_2022_06_29.sqlite')
cursor = conn.cursor()
# Query to select distinct prompts from the generations table
query = "SELECT DISTINCT prompt FROM generations"

# Execute the query and fetch all results
cursor.execute(query)
unique_prompts = cursor.fetchall()

# Close the connection
conn.close()

# Function to clean each prompt
def clean_prompt(prompt):
    # Remove non-ASCII characters
    ascii_prompt = ''.join(char for char in prompt if ord(char) < 128)
    # Replace newlines with space, strip leading/trailing whitespace
    cleaned_prompt = ascii_prompt.replace('\n', ' ').strip()
    return cleaned_prompt

# Write the unique prompts to a text file, applying cleaning
with open('unique_prompts.txt', 'w') as file:
    for prompt in unique_prompts:
        cleaned_prompt = clean_prompt(prompt[0])
        # Write prompt to file if it is not empty or only whitespace
        if cleaned_prompt:
            file.write(cleaned_prompt + '\n')