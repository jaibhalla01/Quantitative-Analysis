import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Title for the app
st.title('Black-Scholes Option Pricing Model')

# Sidebar inputs for model parameters
with st.sidebar:
    st.header('Model Parameters')
    current_asset_price = st.number_input('Current Asset Price', min_value=0.0, step=0.1, format='%.2f', value=50.0)
    strike_price = st.number_input('Strike Price', min_value=0.0, step=0.1, format='%.2f', value=100.0)
    time_to_maturity = st.number_input('Time to Maturity (years)', min_value=0.0, step=0.1, format='%.2f', value=1.0)
    volatility = st.number_input('Volatility', min_value=0.0, max_value=1.05, step=0.05, format='%.2f', value=0.2)
    risk_free_interest_rate = st.number_input('Risk Free Interest Rate', min_value=0.0, max_value=1.05, step=0.05, format='%.2f', value=0.1)

    st.header('Heat Map Parameters')
    hmap_volatility = st.slider('Volatility', min_value=0.0, max_value=1.0, step=0.05, format='%.2f', value=0.5)
    hmap_asset_price = st.slider('Current Asset Price', min_value=0.0, max_value=200.0, step=10.0, format='%.2f', value=100.0)
    hmap_strike_price = st.slider('Strike Price', min_value=0.0, max_value=200.0, step=10.0, format='%.2f', value=100.0)

# Set up the grid for the heatmap
vol_dy = 0.05
asset_dx = 10

# Ensure volatility values remain between 0 and 1
vol_y = np.clip(np.arange(hmap_volatility, hmap_volatility + 8 * vol_dy, vol_dy), 0, 1)
asset_x = np.arange(hmap_asset_price, hmap_asset_price + 8 * asset_dx, asset_dx)

# Black-Scholes pricing model function
def hmap_calculating_options(current_asset_price, volatility, strike_price, time_to_maturity, risk_free_interest_rate):
    d_1 = (np.log(current_asset_price / strike_price) + (risk_free_interest_rate + (volatility ** 2) / 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d_2 = d_1 - volatility * np.sqrt(time_to_maturity)

    call_option_price = stats.norm.cdf(d_1) * current_asset_price - stats.norm.cdf(d_2) * strike_price * np.exp(-risk_free_interest_rate * time_to_maturity)
    put_option_price = strike_price * np.exp(-risk_free_interest_rate * time_to_maturity) * stats.norm.cdf(-d_2) - current_asset_price * stats.norm.cdf(-d_1)

    return call_option_price, put_option_price

def heatmap_axis_formatter(ax):
    # format text labels
    fmt = '{:0.2f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels += [item]

    return xticklabels, yticklabels

# Function to plot the heatmap
def plot_heatmap():
    # input purchase price of the call and put and build a heatmap depicting the P/L of the option
    st.header('Option P/L Heatmaps')

    # Calculate heatmap option prices
    call_profit = np.zeros((len(vol_y), len(asset_x)))  # Adjusting the matrix dimensions to (vol_y, asset_x)
    put_profit = np.zeros((len(vol_y), len(asset_x)))  # Adjusting the matrix dimensions to (vol_y, asset_x)

    for i, y in enumerate(vol_y):
        for j, x in enumerate(asset_x):
            call_price, put_price = hmap_calculating_options(x, y, hmap_strike_price, time_to_maturity, risk_free_interest_rate)
            intrinsic_value_call = max(0, x - hmap_strike_price)
            intrinsic_value_put = max(0, hmap_strike_price - x)
            call_profit[i, j] = intrinsic_value_call - call_price  # Note: Changed to call_profit[i, j] to match (vol_y, asset_x)
            put_profit[i, j] = intrinsic_value_put - put_price    # Note: Changed to put_profit[i, j] to match (vol_y, asset_x)

    # Plotting the call option prices heatmap
    fig, ax = plt.subplots()
    sns.heatmap(call_profit, annot=True, fmt='.2f', xticklabels=np.round(asset_x, 2), yticklabels=np.round(vol_y, 2), ax=ax)
    ax.set_ylim(len(vol_y), 0)  # Ensure correct orientation
    call_x_labels, call_y_labels = heatmap_axis_formatter(ax)
    ax.set_xticklabels(call_x_labels)
    ax.set_yticklabels(call_y_labels)
    ax.tick_params(axis='x', labelrotation=90)

    ax.set_title('Call Option Prices Heatmap')
    ax.set_xlabel('Current Asset Price')
    ax.set_ylabel('Volatility')
    st.pyplot(fig)

    # Plotting the put option prices heatmap
    fig, ax = plt.subplots()
    sns.heatmap(put_profit, annot=True, fmt='.2f', xticklabels=np.round(asset_x, 2), yticklabels=np.round(vol_y, 2), ax=ax)
    ax.set_ylim(len(vol_y), 0)  # Ensure correct orientation
    put_x_labels, put_y_labels = heatmap_axis_formatter(ax)
    ax.set_xticklabels(put_x_labels)
    ax.set_yticklabels(put_y_labels)
    ax.tick_params(axis='x', labelrotation=90)

    ax.set_title('Put Option Prices Heatmap')
    ax.set_xlabel('Current Asset Price')
    ax.set_ylabel('Volatility')
    st.pyplot(fig)

# Display calculated option prices
call_option_price, put_option_price = hmap_calculating_options(current_asset_price, volatility, strike_price, time_to_maturity, risk_free_interest_rate)
st.write(f"Call Option Price: {call_option_price:.2f}")
st.write(f"Put Option Price: {put_option_price:.2f}")

# Plot the heatmaps
plot_heatmap()  # Called the plot_heatmap() function here to update heatmap dynamically
