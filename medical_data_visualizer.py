import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("medical_examination.csv")

df['gluc'].replace({1:0,(2,3):1},inplace= True)
df['cholesterol'].replace({1:0,(2,3):1},inplace= True)
df.gluc.unique()

df['BMI'] = df['weight']/(df['height']/100) ** 2
df['overweight'] = 0

for i in range(len(df['BMI'])):
    if  df['BMI'].iloc[i] > 25:
        df.iloc[i, df.columns.get_loc('overweight')] = 1
df.drop(columns = 'BMI',inplace= True)

def draw_cat_plot():
  df_c0 = df.query('cardio == 0')
  df_c1 = df.query('cardio == 1')

  df_melt_c0 = df_c0 .melt(value_vars=df_c1 [['smoke','gluc','cholesterol','alco','active','overweight']].columns)
  df_melt_c1 = df_c1 .melt(value_vars=df_c1 [['smoke','gluc','cholesterol','alco','active','overweight']].columns)

  df_melt_c0.sort_values('variable', inplace=True)
  df_melt_c1.sort_values('variable', inplace=True)

  dfs = [df_melt_c0,df_melt_c1]


  fig, axs = plt.subplots(1, 2,figsize=(15,6), sharey=True, gridspec_kw={'wspace': 0.05})

  for i in range(2):
    sns.countplot(x = dfs[i]['variable'],hue=dfs[i]['value'],ax=axs[i],)
    axs[i].legend([], frameon=False)
    axs[i].set_title(f'Cardio = {i}')
    axs[i].label_outer()
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)

  axs[0].set(ylabel='total')

  fig.legend([0,1],loc =(0.93,0.5),title='Value',frameon=False,fontsize=14, title_fontsize=14)

  fig.savefig('catplot.png')
  return fig


def draw_heat_map():
  df['pressure'] = df['ap_lo'] <= df['ap_hi']
  df['height_less'] = df['height'] >= df['height'].quantile(0.025)
  df['height_more'] = df['height'] <= df['height'].quantile(0.97)
  df['weight_less'] = df['weight'] >= df['weight'].quantile(0.025)
  df['weight_more'] = df['weight'] <= df['weight'].quantile(0.97)

  lista = ['pressure','height_less','height_more','weight_less','weight_more']
  for i in lista:
    t =  df[(df[i]== False)].index
    df.drop(t, inplace=True)
    df.drop(columns=i, inplace=True)

  corr = df.corr()
  corr= round(corr,1)

  fig, ax = plt.subplots(figsize=(10, 10))
  mask = np.triu(np.ones_like(corr))

  ax =sns.heatmap(corr,  annot=True,
            mask=mask,
            vmax=0.69,vmin=-0.19,
            square=True,
            linewidths=1,
            cmap= 'icefire')

  fig.savefig('heatmap.png')
  return fig