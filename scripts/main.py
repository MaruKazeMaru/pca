from .get_pcs import pca_result

if __name__ == "__main__":
    data_size = int(input())
    if data_size <= 0:
        raise ValueError("data_size <= 0")
    
    datas:list[list[float]] = []
    for i in range(data_size):
        data_l = input().split(' ')
        datas.append([float(s) for s in data_l])

    res = pca_result(datas)
    for
