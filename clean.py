import re

# Specify the file paths
input_file_path = 'data/website_content.txt'
output_file_path = 'data/clean.txt'

# Define the text to be removed as a list
remove_texts = [
    "Why BotPenguin",
    "Product",
    "Solutions",
    "Partners",
    "Resources",
    "The page you were looking for doesn't exist. You may have mistyped the address or the page may have moved."
]

# Define the regular expression pattern to remove sections containing URLs and placeholders
section_pattern = r"================================================================================\n\n--- URL: .+? ---\nHeadings:\nContent:\n  -\n  -\n  -\n  -\n  -\n  -\n\n================================================================================\n"

# Read the input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Remove the specified texts from the content
for text in remove_texts:
    content = re.sub(re.escape(text), '', content)

# Remove sections that match the regular expression pattern
content = re.sub(section_pattern, '', content, flags=re.DOTALL)

# Write the updated content to the output file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("Specified texts and sections have been removed from the file.")
