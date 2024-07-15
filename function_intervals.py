import numpy as np
import matplotlib.pyplot as plt

def cubic():
    '''
    Generate a random cubic function of the form f(x) = a + bx + cx^2 + dx^3
    '''
    rng = np.random.default_rng()
    coeffs = rng.uniform(-10,10,4)
    return lambda x: coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3

def generate_data(f, sigma):
    '''
    Generates random data points from a function f(x) = a + bx + cx^2 + dx^3 by adding random noise to the function
    '''
    rng = np.random.default_rng()

    x = np.linspace(-1,1,100)

    
    y = f(x) + rng.normal(0,sigma,100)
    return x,y

def best_fit(x,y):
    '''
    Finds the best fit line for the data points (x,y) using least squares regression
    '''
    A = np.array([np.ones(len(x)), x, x**2, x**3]).T
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda x: beta[0] + beta[1]*x + beta[2]*x**2 + beta[3]*x**3

sigma = 2
f = cubic()
x,y = generate_data(f,sigma)
g = best_fit(x,y)

# Now we work out a confidence interval for each prediction g(x)
A = np.array([np.ones(len(x)), x, x**2, x**3]).T
cov_g = sigma**2 * A @ np.linalg.inv(A.T @ A) @ A.T
err_y = np.sqrt(np.diag(cov_g))

# Another method: estimate interval for coefficients and then consider corresponding funcition


# Plot results
fig, ax = plt.subplots()
ax.scatter(x,y,s=8, label='Data points')
ax.plot(x,f(x),'r', label='True function')
ax.plot(x,g(x), 'g', label='Best fit curve')
ax.plot(x,g(x) - 2*err_y, 'g--', label='95% confidence interval')
ax.plot(x,g(x) + 2*err_y, 'g--')

# Plot residuals
# for X, Y in zip(x, y):
#     plt.plot([x, x], [y, g(x)], color='blue', linestyle='dashed', linewidth=0.5)

plt.legend()
plt.show()
