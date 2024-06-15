from datasets.factory import Datasets




if __name__ == '__main__':

    # 测试直接复制来的dataset类能不能用
    dataset = "mot17_train_FRCNN"
    datasets = Datasets(dataset)

    for seq in datasets:
        print(f'Found seq: {seq}')
        