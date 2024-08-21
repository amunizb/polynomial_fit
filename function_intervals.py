import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2


def sample_points_from_sphere(n, num_points):
    """
    Sample points uniformly from an n-dimensional sphere.
    
    Parameters:
    n (int): The dimension of the sphere.
    num_points (int): The number of points to sample.
    
    Returns:
    np.ndarray: An array of shape (num_points, n) containing the sampled points.
    """
    # Step 1: Generate points from a standard normal distribution
    points = np.random.randn(num_points, n+1)
    
    # Step 2: Normalize the points to lie on the surface of the n-dimensional sphere
    norms = np.linalg.norm(points, axis=1)
    sphere_points = points / norms[:, np.newaxis]
    
    return sphere_points

def polynomial():
    '''
    Generate a random cubic function of the form f(x) = a + bx + cx^2 + dx^3
    '''
    coeffs = rng.uniform(-1,1,n+1)
    coeffs = np.array(coeffs)

    return lambda x: np.dot(coeffs, [x**i for i in range(n+1)])

def generate_data(f, sigma):
    '''
    Generates random data points from a function f(x) = a + bx + cx^2 + dx^3 by adding random noise to the function
    '''
    num_points = 100

    x = np.linspace(-2,2,num_points)
    y = f(x) + rng.normal(0,sigma,num_points)

    return x,y

def best_fit(x,y):
    '''
    Finds the best fit line for the data points (x,y) using least squares regression
    '''
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda x: np.dot(beta, [x**i for i in range(n+1)]), beta

def confidence_intervals():
    # Now we work out a confidence interval for each prediction g(x)
    cov_g = sigma**2 * A @ np.linalg.inv(A.T @ A) @ A.T
    err_y = np.sqrt(np.diag(cov_g))

    # Another method: estimate interval for coefficients and then consider corresponding function
    # Sample points from sphere
    num_points = 1000  # Number of points to sample
    points = sample_points_from_sphere(n, num_points)

    # Scale points to sphere of radius r
    p = 0.95
    r = sigma * np.sqrt(chi2.ppf(p, n+1))
    points_r = r * points

    # Find Cholesky decomposition 
    U_T = np.linalg.cholesky(A.T @ A)
    U = U_T.T
    U_inv = np.linalg.inv(U)

    #Get limit values for beta
    beta_limit = [beta + U_inv @ point for point in points_r]

    # Find images under this betas
    y_limit = []
    for b in beta_limit:
        y_limit.append(np.dot(A, b))

    # Tranpose y_limit to get all images of each x in a single vector
    y_limit = np.array(y_limit).T
    y_limit_min = np.min(y_limit, axis=1)
    y_limit_max = np.max(y_limit, axis=1)

    return err_y, y_limit_min, y_limit_max


def plot_results():
    # Plot results
    fig, ax = plt.subplots()
    ax.scatter(x,y,s=8, label='Data points')
    ax.plot(x,f(x),'r', label=f'True function')
    ax.plot(x,g(x), 'g', label='Best fit curve')
    ax.fill_between(x, g(x) - 1.96*err_y, g(x) + 1.96*err_y, color='g', alpha=0.3, label='95% confidence interval (point-wise)')
    ax.fill_between(x, y_limit_min, y_limit_max, color='g', alpha=0.1, label='95% confidence interval (coefficients)')

    # Plot residuals
    for X, Y in zip(x, y):
        plt.plot([x, x], [y, g(x)], color='blue', linestyle='dashed', linewidth=0.5)

    plt.legend()
    plt.grid()
    plt.title(f'Best approximation to a polynomial of degree {n}')
    plt.show()

rng = np.random.default_rng()
#n = rng.integers(1,10) # Degree of the polynomial
n=3

sigma = 2
f = polynomial()
x,y = generate_data(f,sigma)
A = np.array([x**i for i in range(n+1)]).T
g, beta = best_fit(x,y)
err_y, y_limit_min, y_limit_max = confidence_intervals()  

plot_results()


# QUESTIONS
# Why don't I see the sample points falling into g(x)\pm 2*err_y?
## The key is that E[\hat Y] = E[A\hat\beta] = A\beta = E[Y/x], and this is NOT the same as y_i
## So what we expect is that 95% of the times the *expected* value of Y for a given x––that is, f(x)––will fall into the interval