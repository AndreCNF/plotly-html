# # Plotly HTMLs
# ---
#
# Notebook where I create plotly plots in HTML, so as to embed in some web page.

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
import yaml                                # Save and load YAML files
import pandas as pd                        # Pandas to load and handle the data
# import data_utils as du                    # Generic data science and machine learning tools
import plotly.graph_objects as go          # Plotly for interactive and pretty plots
import plotly.io as pio                    # Save Plotly graphs
import plotly.express as px
import pycountry

# ## Plotting

# ### Medscape burnouts
#
# https://www.medscape.com/slideshow/2019-global-burnout-comparison-6011180#4

# #### Data

x_data = [22, 27, 28, 12, 38, 37]
y_data = ['UK', 'US', 'France', 'Germany', 'Portugal', 'Spain']

# #### Plot configuration

font = 'Roboto'
font_size = 20
font_color = '#ffffff'
background_color = '#2f528f'
bar_color = '#ffffff'
x_suffix = '%'

# #### Plot

fig = go.Figure()
fig.add_trace(go.Bar(
    y=y_data,
    x=x_data,
    orientation='h',
    marker=dict(
        color=bar_color
    )
))
fig.update_layout(
    title='Percentage of burned out physicians',
    font=dict(
        family=font,
        size=font_size,
        color=font_color
    ),
    paper_bgcolor=background_color,
    plot_bgcolor=background_color,
    xaxis=dict(
        ticksuffix=x_suffix
    ),
    yaxis=dict(
        categoryorder='category descending'
    )
)
fig

pio.write_html(fig, file='medscape_burnouts.html', auto_open=True)

# ### Thesis model component impact
#
# Measuring the average gain in performance that we get from the components of bidirectionality, embedding layer and time awareness.

# #### Data

# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the metrics
metrics_path = 'GitHub/FCUL_ALS_Disease_Progression/metrics/aggregate/'

metrics_files = os.listdir(metrics_path)
try:
    metrics_files.remove('.DS_Store')
except:
    pass
metrics_files

# Create a dictionary with all the metrics:

metrics = dict()
for file_name in metrics_files:
    # Load the current metrics file
    stream = open(f'{metrics_path}{file_name}', 'r')
    model_metrics = yaml.load(stream, Loader=yaml.FullLoader)
    # Remove the extension from the name
    file_name = file_name.split('.yml')[0]
    # Define the model name which will appear in the table
    model_name = ''
    if 'bidir' in file_name:
        model_name = 'Bidirectional '
    if 'tlstm' in file_name:
        model_name += 'TLSTM'
    elif 'mf1lstm' in file_name:
        model_name += 'MF1-LSTM'
    elif 'mf2lstm' in file_name:
        model_name += 'MF2-LSTM'
    elif 'lstm' in file_name:
        model_name += 'LSTM'
    elif 'rnn' in file_name:
        model_name += 'RNN'
    elif 'xgb' in file_name:
        model_name += 'XGBoost'
    elif 'logreg' in file_name:
        model_name += 'Logistic Regression'
    elif 'svm' in file_name:
        model_name += 'SVM'
    if 'embed' in file_name:
        model_name += ', embedded'
    if 'delta_ts' in file_name:
        model_name += ', time aware'
    # Create a dictionary entry for the current model
    metrics[model_name] = dict()
    metrics[model_name]['Avg. Test AUC'] = model_metrics['test']['AUC']['mean']
    metrics[model_name]['Std. Test AUC'] = model_metrics['test']['AUC']['std']

# Convert to a dataframe:

metrics_df = pd.DataFrame(metrics)
metrics_df

# Transpose to have a row per model:

metrics_df = metrics_df.transpose()
metrics_df

# Sort by a descending order of performance:

metrics_df = metrics_df.sort_values('Avg. Test AUC', ascending=False)
metrics_df

model_names = list(metrics_df.index)
model_names

component_gains = dict()
components_str = dict(bidirectionality='Bidirectional ', 
                      embedding=', embedded', 
                      time_awareness=', time aware')
for component in components_str.keys():
    # Find and match the names of the models with and without the component
    models_without_comp = [model_name.replace(components_str[component], '') 
                           for model_name in model_names 
                           if components_str[component] in model_name]
    models_with_comp = [model_name 
                        for model_name in model_names 
                        if components_str[component] in model_name]
    model_comp_names_match = dict(zip(models_without_comp, models_with_comp))
    curr_component_gains = list()
    for model_name in models_without_comp:
        # Calculate the difference in model performance with and without the component
        component_gain = (metrics_df.loc[model_comp_names_match[model_name], 'Avg. Test AUC'] 
                          - metrics_df.loc[model_name, 'Avg. Test AUC'])
        curr_component_gains.append(component_gain)
    # Average the component's effect
    component_gains[component] = sum(curr_component_gains) / len(curr_component_gains)
component_gains

# Find and match the names of the models with LSTM and with RNN
models_with_lstm = [model_name.replace('RNN', 'LSTM')
                    for model_name in model_names 
                    if 'RNN' in model_name]
models_with_rnn = [model_name 
                   for model_name in model_names 
                   if 'RNN' in model_name]
model_comp_names_match = dict(zip(models_with_rnn, models_with_lstm))
curr_component_gains = list()
for model_name in models_with_rnn:
    # Calculate the difference in model performance with LSTM and with RNN
    component_gain = (metrics_df.loc[model_comp_names_match[model_name], 'Avg. Test AUC'] 
                      - metrics_df.loc[model_name, 'Avg. Test AUC'])
    curr_component_gains.append(component_gain)
# Average LSTM's effect
component_gains['LSTM'] = sum(curr_component_gains) / len(curr_component_gains)
component_gains

# Convert to a dataframe:

gain_df = pd.Series(component_gains, name='Avg. Impact on Test AUC')
gain_df

gain_df.index = ['Bidirectionality', 'Embedding', 'Time Awareness', 'LSTM']
gain_df

gain_df.index.rename('Component')
gain_df

# Sort by a descending order of performance gain:

gain_df = gain_df.sort_values(ascending=False)
gain_df

# #### Plot configuration

font = 'Roboto'
font_size = 20
font_color = '#ffffff'
background_color = '#8f2f2f'
marker_color = ['#FF9999',
                '#99FFFF',
                '#FFFF99',
                '#99FF99']
marker_color.reverse()

# #### Plot

gain_plot_df = gain_df.copy()
gain_plot_df = gain_plot_df.sort_values(ascending=True)
# Create the figure
figure=dict(
    data=[dict(
        type='bar',
        x=gain_plot_df,
        y=gain_plot_df.index,
        orientation='h',
        marker=dict(color=marker_color)
    )],
    layout=dict(
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        title='Average impact on model\'s test AUC',
        yaxis_title=gain_plot_df.index.name,
        font=dict(
            family=font,
            size=font_size,
            color=font_color
        )
    )
)
fig = go.Figure(figure)
fig

pio.write_html(fig, file='GitHub/test-plotly-html/thesis_component_impact.html', auto_open=True)

# ### Thesis bidir LSTM time aware feature importance

# #### Data

# Change to parent directory (presumably "Documents")
os.chdir("../..")

data_path = 'GitHub/hai-dash/data/ALS/'
data_file_name = 'fcul_als_with_shap_for_lstm_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3784valloss_08_07_2020_04_14'

df = pd.read_csv(f'{data_path}{data_file_name}.csv')
df.head()

# Get the SHAP values into a NumPy array and the feature names
shap_column_names = [feature for feature in df.columns
                     if feature.endswith('_shap')]
feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
shap_values = df[shap_column_names].to_numpy()

# #### Plot configuration

font = 'Roboto'
font_size = 20
font_color = 'white'
background_color = '#8f8e2f'
marker_color = 'white'
max_display = 10

# #### Plot

# Generate the SHAP summary plot
fig = du.visualization.shap_summary_plot(shap_values, feature_names,
                                         max_display=max_display,
                                         background_color=background_color,
                                         marker_color=marker_color,
                                         output_type='plotly',
                                         font_family=font, font_size=font_size,
                                         font_color=font_color,
                                         xaxis_title='mean(|SHAP value|)')
fig

pio.write_html(fig, file='GitHub/test-plotly-html/thesis_feat_import_bidir_lstm_delta_t.html', auto_open=True)

# ### Thesis XGBoost feature importance

# #### Data

# Change to parent directory (presumably "Documents")
os.chdir("../..")

data_path = 'Datasets/Thesis/FCUL_ALS/interpreted/'
data_file_name = 'fcul_als_with_shap_for_xgb_0.5926valloss_09_07_2020_02_40'

df = pd.read_csv(f'{data_path}{data_file_name}.csv')
df.head()

# Get the SHAP values into a NumPy array and the feature names
shap_column_names = [feature for feature in df.columns
                     if feature.endswith('_shap')]
feature_names = [feature.split('_shap')[0] for feature in shap_column_names]
shap_values = df[shap_column_names].to_numpy()

# #### Plot configuration

font = 'Roboto'
font_size = 20
font_color = 'white'
background_color = '#8f8e2f'
marker_color = 'white'
max_display = 10

# #### Plot

# Generate the SHAP summary plot
fig = du.visualization.shap_summary_plot(shap_values, feature_names,
                                         max_display=max_display,
                                         background_color=background_color,
                                         marker_color=marker_color,
                                         output_type='plotly',
                                         font_family=font, font_size=font_size,
                                         font_color=font_color,
                                         xaxis_title='mean(|SHAP value|)')
fig

pio.write_html(fig, file='GitHub/test-plotly-html/thesis_feat_import_xgb.html', auto_open=True)

# ### Countries visited
#
# For my [personal page](https://andrecnf.com/).

# #### Data

visited_countries = ['Portugal', 'Spain', 'France', 'Germany', 'Netherlands', 
                     'United Kingdom', 'Croatia', 'Bosnia and Herzegovina', 'Italy', 'Slovenia',
                     'Colombia', 'Peru', 'Chile', 'Argentina', 'Brazil', 
                     'United States', 'Costa Rica', 'China', 'Hong Kong',
                     'Macao']

for country in visited_countries:
    try:
        pycountry.countries.get(name=country).alpha_3
    except:
        print(f'Failed to encode {country}')

country_codes = [pycountry.countries.get(name=country).alpha_3 for country in visited_countries]
country_codes

# #### Plot

fig = px.choropleth(locations=country_codes)
fig.update_layout(dict(
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0)
))

# Change to parent directory (presumably "Documents")
os.chdir("../..")

pio.write_html(fig, file='GitHub/test-plotly-html/countries_visited.html', auto_open=True)


