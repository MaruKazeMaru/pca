import argparse

import principal_component_analysis as pca

def main(args):
    datas = []
    with open(args.datafile, mode="r") as f:
        for line in f:
            data = [float(s) for s in line.split(' ')]
            datas.append(data)

    a = pca.pca_assister(datas)
    result = a.pca_dim(args.dim)

    if args.show:
        print("avr=")
        print(a.avr)
        print("eig_vals=")
        print(a.eig_vals)
        print("eig_vecs=")
        print(a.eig_vecs)

    for i in range(args.dim):
        p:str = args.datafile
        ps = p.split("/")
        fname = ps[-1]
        l = fname.split(".")
        l[0] += "_pca" + str(i+1)
        fname = ".".join(l)
        ps[-1] = fname
        p = "/".join(ps)
        with open(p, "w") as f:
            for j in range(len(result)):
                f.write(str(result[j][i]) + "\n")


if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description="pricial component analysis")
    parser.add_argument("datafile", help="path of data file")
    parser.add_argument("-d", "--dim", help="pca max dimension", type=int, default=1)
    parser.add_argument("-s", "--show", help="show average & eigen values & eigen vectors", action="store_true")
    args = parser.parse_args()
    main(args)
