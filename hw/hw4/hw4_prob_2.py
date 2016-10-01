import pandas as pd


def encode_categorical_variables_prob_2a():
    df = pd.read_csv('datasets/dataset_2.txt')
    print df.columns


def main():
    encode_categorical_variables_prob_2a()


if __name__ == '__main__':
    main()
