1. Initializes random polynomial $p: \R \to \R$ of random degree $n$.
2. Samples data point $(x_i, y_i)$ where $y_i = p(x_i) + \epsilon_i$ for $\epsilon_i \sim \mathcal N(0,\sigma^2)$ i.i.d. normal random variables.
3. Finds best fit $\hat p$ using OLS.
4. Plots $p$, $\hat p$, and both point-wise and coefficient-based confidence intervals for $\hat p$.
