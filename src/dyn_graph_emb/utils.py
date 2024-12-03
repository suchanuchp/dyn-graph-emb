def save_list_to_file(lst, file_path):
    with open(file_path, 'w') as f:
        for elt in lst:
            f.write(f"{elt}\n")
