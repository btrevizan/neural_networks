from matplotlib import pyplot as plt
from src.utils import datasets
import seaborn as sns
import pandas as pd
import numpy as np

params = {'alpha': 'Taxa de Aprendizado',
          'batchsize': 'Tamanho do Batch (porção)',
          'beta': 'Parâmetro do Método do Momento (beta)',
          'nlayers': 'Número de Camadas Ocultas',
          'nneurons': 'Número de neurônios por camada',
          'regularization': 'Parâmetro de Regularização (lambda)'}


def plot_and_save(data, p, y):
    ax = sns.lineplot(x='param_value',
                      y=y,
                      hue='Dataset',
                      style='Dataset',
                      markers=True,
                      dashes=False,
                      data=data)

    ax.set_xlabel(params[p])
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_ylim(0.0, 1.0)

    if p == 'batchsize':
        ax.set_xticks([0.06, 0.125, 0.25, 0.5, 1])
    else:
        ax.set_xticks(np.unique(data['param_value'].values))

    plt.savefig('tests/graphics/{}_{}.pdf'.format(p, y), dpi=360, pad_inches=0.1, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    for p in params:
        p_path = 'tests/results/{}/' + 'cv_{}.csv'.format(p)
        data = None

        for d in datasets:
            d_path = p_path.format(d)
            d_data = pd.read_csv(d_path, header=0)

            if p == 'batchsize':
                d_data['param_value'] = (d_data['param_value'] / d_data.iloc[-1, 0]).round(3)

            seconds = d_data['seconds_elapsed'].values
            time_elapsed = (seconds - np.min(seconds)) / (np.max(seconds) - np.min(seconds))
            time_column = pd.DataFrame({'Time (normalized)': time_elapsed})
            d_data = pd.concat([time_column, d_data], axis=1)

            d_column = pd.DataFrame({'Dataset': [d] * d_data.shape[0]})
            d_data = pd.concat([d_column, d_data], axis=1)

            if data is None:
                data = d_data
            else:
                data = pd.concat([data, d_data], ignore_index=True)

        mean_score = pd.DataFrame(data.iloc[:, 4:].mean(axis=1), columns=['Mean F1-Score'])
        std_score = pd.DataFrame(data.iloc[:, 4:].std(axis=1), columns=['Standard Deviation of F1-Score']).round(2)

        data = pd.concat([data, mean_score, std_score], axis=1)

        plot_and_save(data, p, 'Time (normalized)')
        plot_and_save(data, p, 'Mean F1-Score')
        plot_and_save(data, p, 'Standard Deviation of F1-Score')
