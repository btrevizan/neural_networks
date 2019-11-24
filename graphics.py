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


def plot_and_save(data, p, y, name):
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
        ax.set_xticks([1/16, 1/8, 1/4, 1/2, 1])
    elif p == 'regularization':
        ax.set_xticks([0.01, 1, 10])
    else:
        ax.set_xticks(np.unique(data['param_value'].values))

    plt.savefig('tests/graphics/{}_{}.pdf'.format(p, name), dpi=360, pad_inches=0.1, bbox_inches='tight')
    plt.close()


def plot_costs(param):
    for d in datasets:
        data = pd.read_csv('tests/results/{}/costs_{}.csv'.format(d, param.replace(' ', '').lower()), header=0, index_col=None)

        ax = sns.lineplot(x='n_instance',
                          y='cost',
                          hue=param,
                          style=param,
                          markers=False,
                          dashes=False,
                          data=data)

        ax.set_ylabel('Custo')
        ax.set_xlabel('Número de instâncias apresentadas')

        ax.tick_params(axis='x', which='major', labelsize=4)
        ax.set_yticks(np.linspace(data['cost'].min(), data['cost'].max(), 10))
        ax.set_xticks(np.linspace(data['n_instance'].min(), data['n_instance'].max(), 10))

        plt.savefig('tests/graphics/{}_costs_{}.pdf'.format(d, param.replace(' ', '')),
                    dpi=360,
                    pad_inches=0.1,
                    bbox_inches='tight')

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

            d_column = pd.DataFrame({'Dataset': [d] * d_data.shape[0]})
            d_data = pd.concat([d_column, d_data], axis=1)

            if data is None:
                data = d_data
            else:
                data = pd.concat([data, d_data], ignore_index=True)

        mean_score = pd.DataFrame(data.iloc[:, 3:].mean(axis=1), columns=['Média do F1-Score'])
        std_score = pd.DataFrame(data.iloc[:, 3:].std(axis=1), columns=['Desvio padrão do F1-Score']).round(2)

        data = pd.concat([data, mean_score, std_score], axis=1)

        plot_and_save(data, p, 'Média do F1-Score', 'Mean')
        plot_and_save(data, p, 'Desvio padrão do F1-Score', 'Std')

    plot_costs('Beta')
    plot_costs('Batch size')
    plot_costs('Alpha')
