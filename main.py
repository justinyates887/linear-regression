from numpy import *

def compute_error_for_line_given_points(b, m, points):
    #Error Formula for LR:
    #   Error(m,b) = 1/N Sig N, i=1 (y(sub)i - (mx(sub)i + b))^2

    totalError = 0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]

        totalError += (y - (m * x + b)) ** 2
    
    #Return avg
    return totalError / float(len(points))

def step_gradient(b, m, points, rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        #Direction with respect to b and m
        #Computing partial derivatives of error     
        # d/dm = 2/N Sig N, i = 1 - x(sub)i(y(sub)i - (mx(sub)i + b))
        # d/db = 2/N Sig N, i = 1 - (y(sub)i - (mx(sub)i + b))
        b_gradient += -(2/N) * (y - ((m * x) + b))
        m_gradient += (2/N) * x * (y - ((m * x) + b))

        #Update b and m using partial ders
        new_b = b - (rate * b_gradient)
        new_m = m - (rate * m_gradient)

        return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, rate, iterations):
    b = starting_b
    m = starting_m

    for i in range(iterations):
        #update b and m with new values
        b, m = step_gradient(b, m, array(points), rate)

    return [b,m]

def run():

    #Collect data from CSV (x,y)
    points = genfromtxt('data.csv', delimiter=',')

    # Define hyperparameters (Parameters that define how our model is analyzing data)

    #How fast our model converges
    learning_rate = 0.0001 #balance for compute
    #y = mx+b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000 #small dataset does not require large iter

    #Train model
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_b, points)))
    [b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("Ending gradient descent at b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(initial_b, initial_b, points)))

if __name__ == '__main__':
    run()