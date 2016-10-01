import pandas as pd


def main():
    df = pd.read_csv('datasets/dataset_2.txt')

    print df.columns


if __name__ == '__main__':
    main()
