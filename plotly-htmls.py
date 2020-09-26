# # Plotly HTMLs
# ---
#
# Notebook where I create plotly plots in HTML, so as to embed in some web page.

# ## Importing the necessary packages

import plotly.graph_objects as go
import plotly.io as pio

# ## Plotting

# ### Medscape burnouts
#
# https://www.medscape.com/slideshow/2019-global-burnout-comparison-6011180#4

# Data:

x_data = [22, 27, 28, 12, 38, 37]
y_data = ['UK', 'US', 'France', 'Germany', 'Portugal', 'Spain']

# Plot configuration:

font = 'Roboto'
font_size = 20
font_color = '#ffffff'
background_color = '#2f528f'
bar_color = '#ffffff'
x_suffix = '%'

# Plot:

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


