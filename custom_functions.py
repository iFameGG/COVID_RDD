import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import datetime

def rdd_plot(df, model, ax, title):
    """
    Function made to plot RDD regression models.
    """
    ax.scatter(df['Time'], df['New_Cases'])

    first_reg = df[df.Time < 0]
    new_row = pd.DataFrame(first_reg.iloc[-1, :].to_dict(), index=[len(first_reg)])
    first_reg = pd.concat([first_reg, new_row])
    first_reg.iloc[-1, 2] = first_reg.iloc[-1, 2] + 0.999999
    ax.plot(
        first_reg['Time'], 
        model.predict(first_reg), 
        color='r', 
        linewidth=4, 
        alpha=0.7,
        label='Regression'
    )

    second_reg = df[df.Time > 0]
    new_row = pd.DataFrame(second_reg.iloc[0, :].to_dict(), index=[len(second_reg)])
    second_reg = pd.concat([new_row, second_reg])
    second_reg.iloc[0, 2] = second_reg.iloc[0, 2] - 0.999999
    ax.plot(
        second_reg['Time'], 
        model.predict(second_reg), 
        color='r', 
        linewidth=4, 
        alpha=0.7
    )

    ax.axvline(x=0, color = 'gray', linewidth=5, linestyle='--', label='Cutoff')

    ax.legend()
    title_font = {'fontsize': 15, 'fontweight': 'bold'}
    ax.set_title(title + f'\n(R-Squared: {round(model.rsquared,3)})', fontdict=title_font)
    ax.set_xlabel('Days from the cutoff point', fontsize=14)
    ax.set_ylabel('New Cases', fontsize=14)

def coef_p_table(fig, ax, df, title):
    df = df.round(3)
    ax.axis('off')
    col_names = list(df.columns)
    max_row = 0
    max_col = 0
    for row in range(len(df)):
        if row>0:
            row_size = row/3
        else:
            row_size = row
        row_data = df.iloc[row]
        for col in range(len(row_data)):
            if col>0:
                col_size = col/1.5
            else:
                col_size = col
            ax.text(x=col_size, y=row_size, s=row_data[col_names[col]], va='center', ha='left', fontsize=13)
            max_col = col_size
        max_row = row_size

    header_height = max_row+0.15
    for col_idx in range(len(col_names)):
        if col_idx>0:
                col_size = col_idx/1.5
        else:
            col_size = col_idx
        ax.text(col_size, header_height, col_names[col_idx], weight='bold', ha='left', fontsize=11)

    # header seperator
    ax.plot(
        [0, max_col + 23],
        [max_row+0.1, max_row+0.1],
        c='black'
    )

    ax.set_ylim([0, header_height+0.05])
    ax.set_xlim([0, max_col+0.8])
    title_font = {'fontsize': 16, 'fontweight': 'bold'}
    ax.set_title(title, pad=15, fontdict=title_font)

def coef_p_table2(fig, ax, df, title):
    df = df.round(3)
    ax.axis('off')
    col_names = list(df.columns)
    max_row = 0
    max_col = 0
    for row in range(len(df)):
        if row>0:
            row_size = row/4.5
        else:
            row_size = row
        row_data = df.iloc[row]
        for col in range(len(row_data)):
            if col>0:
                col_size = col/1.5
            else:
                col_size = col
            ax.text(x=col_size, y=row_size, s=row_data[col_names[col]], va='center', ha='left', fontsize=13)
            max_col = col_size
        max_row = row_size

    # Main header
    header_height = max_row+0.2
    for col_idx in range(len(col_names)):
        if col_idx>0:
            column_name = col_names[col_idx][1]
            col_size = col_idx/1.5
        else:
            column_name = col_names[col_idx][0]
            col_size = col_idx
        ax.text(col_size, header_height, column_name, weight='bold', ha='left', fontsize=11)
    
    # cat header
    header_height = max_row+0.4
    for cat_head, offset in zip(['Linear Model', 'Poly Model'], [(col_size/2)-.2, (col_size/2)+1+.2]):
        ax.text(offset, header_height, cat_head, weight='bold', ha='center', fontsize=11)
    
    # vertical seperators
    plt.axvline(x = (col_size/2)+0.5, color = 'black')
    plt.axvline(x = 0.6, color = 'black')

    # header seperator
    ax.plot(
        [0, max_col + 17],
        [max_row+0.15, max_row+0.15],
        c='black'
    )

    ax.set_ylim([0, header_height+0.05])
    ax.set_xlim([0, max_col+0.8])
    title_font = {'fontsize': 16, 'fontweight': 'bold'}
    ax.set_title(title, pad=15, fontdict=title_font)

def diff_rdd_fig(df, model, model_poly, model_info, suptitle):
    plt.close('all')

    gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 2.5], height_ratios=[2, 0.75])

    # create a figure object using the gridspec object
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    # ax4 = fig.add_subplot(gs[2, :])

    # set the position of each subplot using the gridspec object
    ax1.set_position(gs[0, 0].get_position(fig))
    ax2.set_position(gs[0, 1].get_position(fig))
    ax3.set_position(gs[1, :].get_position(fig))
    # ax4.set_position(gs[2, :].get_position(fig))

    rdd_plot(df, model, ax1, 'Linear RDD')
    rdd_plot(df, model_poly, ax2, 'Poly RDD')
    coef_p_table2(fig, ax3, model_info, 'Coefficients and P-Values')
    # coef_p_table(fig, ax3, coefs, 'Coefficients')
    # coef_p_table(fig, ax4, info_m, 'P-Values and R-Squared')

    fig.suptitle(suptitle, y=1.02 , fontsize=20, weight='bold')
    fig.tight_layout(pad=2)

def vax_plot(ax, data):
    vax = pd.read_csv('QC_Vax.csv')
    vax['date'] = pd.to_datetime(vax['date'], format='%Y-%m-%d')
    daily_vax_cols = [
        'Age_0_4_ans_DOSE_Numero1_jour', 'Age_5_11_ans_DOSE_Numero1_jour',
        'Age_12_17_ans_DOSE_Numero1_jour', 'Age_18_24_ans_DOSE_Numero1_jour',
        'Age_25_29_ans_DOSE_Numero1_jour', 'Age_30_34_ans_DOSE_Numero1_jour',
        'Age_35_39_ans_DOSE_Numero1_jour', 'Age_40_44_ans_DOSE_Numero1_jour',
        'Age_45_49_ans_DOSE_Numero1_jour', 'Age_50_54_ans_DOSE_Numero1_jour',
        'Age_55_59_ans_DOSE_Numero1_jour', 'Age_60_64_ans_DOSE_Numero1_jour',
        'Age_65_69_ans_DOSE_Numero1_jour', 'Age_70_74_ans_DOSE_Numero1_jour',
        'Age_75_79_ans_DOSE_Numero1_jour', 'Age_80_84_ans_DOSE_Numero1_jour',
        'Age_85_110_ans_DOSE_Numero1_jour'
    ]
    vax_daily = vax[daily_vax_cols].sum(axis=1)
    vax_data = pd.concat([vax.date, vax_daily], axis=1)
    vax_data.columns = ['date', 'vaccinations']
    vax_plot_data = vax_data.loc[
        (vax_data['date'] > '2020-12-10')
        & 
        (vax_data['date'] < '2021-1-30')
        ]

    plot_data = data.loc[
        (data['Date'] > '2020-12-10')
        & 
        (data['Date'] < '2021-1-30')
        ]

    ax.axvline(datetime.datetime(2020, 12, 25), color='red', label='2020-12-25 lockdown', linewidth=3) # Lockdown 2
    ax.axvspan(datetime.datetime(2020, 12, 25), datetime.datetime(2020, 12, 31), alpha=0.3, color='red') # incubation period
    ax.axvline(datetime.datetime(2020, 12, 14), color='green', label='First Vaccinations', linewidth=3) # First Vaccinations
    ax.axvline(datetime.datetime(2021, 1, 9), color='gray', label='Cutoffs', linewidth=4)

    ax.bar(vax_plot_data['date'], vax_plot_data['vaccinations'], color='cyan', label='New Vaccinations')
    ax.plot(plot_data['Date'], plot_data['New_Cases'], label='New Cases', linewidth=3)

    ax.legend()
    ax.set_ylim(0, 3500)
    ax.set_xlabel('Date')
    ax.set_title('COVID Cases and Vaccinations')

def overview_plot(ax, data):
    # Zoom into the portion of the data that is part of the project
    plot_data = data.loc[
        (data['Date'] > '2020-3-10')
        & 
        (data['Date'] < '2021-2-25')
        ]

    # Build the scatter plot
    ax.scatter(plot_data['Date'], plot_data['New_Cases'])

    #### Important Dates to Check
    ax.axvline(datetime.datetime(2020, 3, 20), color='orange', label='2020-03-20 lockdown') # Lockdown 1
    ax.axvspan(datetime.datetime(2020, 3, 20), datetime.datetime(2020, 3, 26), alpha=0.3, color='orange') # incubation period
    ax.axvline(datetime.datetime(2020, 4, 14), color='gray')

    ax.axvline(datetime.datetime(2020, 8, 31), color='gold', label='2020-08-31 reopening of schools') # Reopen Schools
    ax.axvspan(datetime.datetime(2020, 8, 31), datetime.datetime(2020, 9, 8), alpha=0.3, color='gold') # incubation period
    ax.axvline(datetime.datetime(2020, 9, 20), color='gray')

    ax.axvline(datetime.datetime(2020, 12, 25), color='red', label='2020-12-25 lockdown') # Lockdown 2
    ax.axvspan(datetime.datetime(2020, 12, 25), datetime.datetime(2020, 12, 31), alpha=0.3, color='red') # incubation period
    ax.axvline(datetime.datetime(2021, 1, 9), color='gray', label='Cutoffs')

    # ax.axvline(datetime.datetime(2020, 12, 14), color='green', label='First Vaccinations') # First Vaccinations

    # Format the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('New Cases')
    ax.set_title('New Cases Per Day')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax.grid(True, 'major', axis='y', color='black',alpha=0.3,linestyle='--')
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    myFmt = mdates.DateFormatter('%B %dst %Y')
    ax.xaxis.set_major_formatter(myFmt)