import os


def get_root_project_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


def get_word_list():
    word_list = []
    with open("./data/words_list.txt", "r") as input:
        for line in input.readlines():
            word_list.append(line.replace("\n", ""))
    return word_list
