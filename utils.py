def read_and_split_content(filename):
    with open("./data/" + filename, "r", encoding="utf-8") as f:
        content = f.read()

        # Check if the content should be split into sentences or paragraphs
        if "mouse" in filename:
            # Split by each new line for mouse files
            data = content.split("\n")
        else:
            # Splitting by two new lines to get paragraphs for other files
            data = content.split("\n\n")

        half_length = len(data) // 2
        return (
            data[half_length:],
            data[:half_length],
        )  # Return test_data, training_data


def extract_categories_from_filenames(filename1, filename2):
    # Extract category names from the filenames
    category1 = filename1.split(".")[0]
    category2 = filename2.split(".")[0]
    return category1, category2
