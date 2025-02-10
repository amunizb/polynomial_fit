# Polynomial fit using OLS
1. Initializes random polynomial $p: \mathbb R \to \mathbb R$ of random degree $n$.
2. Samples data point $(x_i, y_i)$ where $y_i = p(x_i) + \epsilon_i$ for $\epsilon_i \sim \mathcal N(0,\sigma^2)$ i.i.d. normal random variables.
3. Finds best fit $\hat p$ using OLS.
4. Plots $p$, $\hat p$, prediction bars $y_i - \hat y_i$ and both point-wise and coefficient-based confidence intervals for $\hat p$.
