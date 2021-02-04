import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for i in range(1,9):
        df = pd.read_csv(f"out/img{i}.csv", index_col=0)
        plt.plot(df,marker='o', label='image '+str(i))
        plt.ylabel('PSNR')
        plt.xlabel('compression degree')
        plt.legend()
    plt.savefig(f"out/plot.jpg")
