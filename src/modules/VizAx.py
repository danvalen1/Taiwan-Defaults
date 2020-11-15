import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


labels_dict = {'pay_status_apr': 'Payment Status in April'}
figsize = (5, 4)
dpi = (300)
fontscale = 1.2
sns.set(font_scale = fontscale, style = 'whitegrid')

def PlotScatter(df, xvar, yvar, targetdir, hue=None):
    """Plots a scatter of `xvar` and `yvar` in df. A hue can be added through `hue`. 
    """
    
    title = f'{labels_dict[yvar]} v. {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                      
    scatter = sns.scatterplot(x=xvar,
                    y=yvar,
                    data=df,
                    hue=hue,
                    palette="Spectral",
                    s=markersize,
                    alpha = .3
                   )
    if hue:
        ax.legend([hue])
        ax.legend(markerscale=1.5)
        title = f'{labels_dict[yvar]} v. {labels_dict[xvar]}\n Colored By {hue.title()} Category'
    
    ax.set(title=f'{title}',
          xlabel=labels_dict[xvar],
          ylabel=labels_dict[yvar]
          )
    
    
        
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    if xvar=='sqft_lot':
        plt.xticks(rotation=-45)
    
    fig.savefig(f'{targetdir}{title}.png', bbox_inches='tight')
                        
    return plt.show()
    
def PlotHist(df, xvar, targetdir, bins='auto'):
    """Plot a histogram with automatic labels provided in a global dict in CustomModule. 
        Pass a dataframe through `df`, a string through `xvar`, and the number of bins through `bins`.
    """
    
    title = f'Frequency of {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    
    sns.histplot(data=df,
                x=xvar,
                bins=bins)
    
    ax.set(title=title,
          xlabel=labels_dict[xvar],
          ylabel='Frequency'
          )
    
    plt.locator_params(axis='x', nbins=bins)
    
    
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    
    fig.savefig(f'{targetdir}{title}.png', bbox_inches='tight')
    
    return plt.show()

def CorrHeatmap(df, targetdir):
    """Generates a correlation heatmap from a dataframe with variables of interest.
    """
    fig, ax = plt.subplots(figsize=(40,32), dpi=300)
    sns.heatmap(df.corr(), cmap='bwr', center=0, annot=True)
    fig.savefig(f'{targetdir}CorrHeatmap.png', bbox_inches='tight')
    return fig.show()

def PlotCMatrix(cmatrix):
    fig, ax = plt.subplots(figsize = figsize)
    sns.heatmap(cmatrix, annot=True, fmt='g', ax=ax, cmap='Blues')
    ax.set_xticklabels(['Employed', 'Unemployed'])
    ax.set_yticklabels(['Employed', 'Unemployed'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    ax.set_title('Confusion Matrix Using L1 regularized Logistic Regression')
    return plt.show()
