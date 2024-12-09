def save_list_to_file(lst, file_path):
    with open(file_path, 'w') as f:
        for elt in lst:
            f.write(f"{elt}\n")


def read_list_from_file(file_path):
    lst = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            lst.append(line.rstrip())
    return lst

