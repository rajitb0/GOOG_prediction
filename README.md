# Recurrent Interpolants for GOOG stock Forecasting

## How To Run?

Download notebook from here directly (No cloning required) or copy code. Then upload notebook/code onto a Jupyter Notebook environment (Google Colab, Vscode, etc.) and run all cells in order. No scripts or external data required since everything is extracted in notebook.

## Overview
(Disclaimer: Google Gemini 3 was used as coding assistance for the Recurrent Interpolation and LSTM aspect of this project)

This project explores **Recurrent Interpolants (RI)** as a method for probabilistic time series prediction and compares their performance against a standard **LSTM baseline** on stock price data.

Recurrent Interpolants are a recently proposed approach that combines ideas from diffusion models, stochastic differential equations, stochastic interpolants and recurrent neural networks. They are designed to model conditional distributions of future observations given past context.

Reference: Chen, Yu, et al. “Recurrent Interpolants for Probabilistic Time Series Prediction.” arXiv preprint arXiv:2409.11684, 2024. [https://arxiv.org/abs/2409.11684].

The goal of this project was to compare how Recurrent Interpolants behave and whether their probabilistic formulation provides an advantage over simpler sequence models such as LSTMs for long term horizon forecasting.

---

## What are Recurrent Interpolants?

Recurrent Interpolants model the transition from one time step to the next as a **stochastic interpolation** between the current observation and the future observation.

At a high level:

* An RNN encodes the past context into a hidden state.
* A stochastic interpolant is constructed between the current value and the next value using a time parameter ( s \in [0,1] ) and injected noise.
* Two neural networks are trained:

  * A velocity network, which learns the drift of the interpolant.
  * A score network, which learns the gradient of the log-density.
* Training minimizes losses derived from stochastic calculus identities.
* Inference can be performed by integrating a learned stochastic differential equation forward in time.

Recurrent Interpolants aim to learn the full conditional distribution of the next time step.

---

## Implementation

* Downloaded historical GOOG daily stock (Google stock) price data from somewhere between Jun to Aug 2004 to Dec 5 2025 from Yahoo Finance.
* Performed feature engineering (OHLC averages, moving averages, volatility, percentile rank).
![alt text](https://github.com/rajitb0/GOOG_prediction/blob/main/stock.png "")
* Trained two models:

  1. A Recurrent Interpolant model with an RNN encoder and learned interpolant dynamics.
  2. A simple LSTM model trained to predict the next closing price.
* Evaluated both models using rolling one-step-ahead predictions.
* Visualized the last 200 points of the time series with model forecasts overlaid.

* RNN and LSTM both used MSE loss.
* 5 epochs was used to train LSTM with single layer with a depth of 32 nodes.
  

---

## Results

In this setting, the Recurrent Interpolant model performed noticeably worse than the LSTM:

![alt text](https://github.com/rajitb0/GOOG_prediction/blob/main/results_ri.png "")

* RI predictions captured the overall long term trend but completely messed up the prices.
* The LSTM tracked both overall price movement and trends much more closely.

These results were consistent across multiple rolling windows (20 and 50 day rolling window).

---

## Why Recurrent Interpolants underperformed here

There are several reasons why Recurrent Interpolants struggled in this experiment:

1. Optimization difficulty
   RI models are significantly harder to train. They require learning multiple networks simultaneously under stochastic objectives. With limited tuning and relatively short training, the model often fails to learn properly.

2. High Variance
   According to the paper, their loss functions (score + velocity) experiences high variance when close to either distributions. They recommend importance sampling to overcome this issue.

4. Limited data efficiency
   Diffusion-style methods typically require large datasets and long training runs. Daily stock data even over many years is still small (5000 samples).

5. Implementation simplifications
   This project used simple inference and not full SDE sampling over multiple steps. This removes much of the advantage of RI.

---

## Takeaway

Implementation of Recurrent Interpolant was most likely not right due to gap in conceptual knowledge.

Even with proper implementation, it is expected that:

For short-horizon stock forecasting:

* Recurrent Interpolants may perform better when implemented but... 

For long term stock forecasting:

* LSTM will likely perform better as seen due to its memory cell architecture that allows it to retain information over long periods and thus successfully identifies long term patterns. 

---

## Files

* `Stoch_Interpol_ECE69500.ipynb`: Main notebook containing data processing, model definitions, training loops, and visualizations.
