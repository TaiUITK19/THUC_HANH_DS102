import numpy as np

def grad(u):
        x, y = u
        return np.array([
                4*x**3 + 4*x*y,
                4*y**3 + 2*x**2
        ])


def hessian(u):
        x, y = u
        return np.array([
                [
                        12*x**2 + 4*y, 
                        4*x
                 ],
                [
                        4*x, 
                        12*y**2
        ]
        ])


def newton_optim(u_0, grad_fn, hessian_fn, epsilon=10e-8, max_iter=100):
        u = u_0
        for i in range(max_iter):
                grad = grad_fn(u)


                if np.linalg.norm(grad) < epsilon:
                        return u


                hess = hessian_fn(u)

                # t = grad @ np.linalg.inv(hessian)
                t = np.linalg.solve(hess, -grad)

                u = u + t
        return u

if __name__ == "__main__":
        u_0 = (10, 10)
        u_optim = newton_optim(u_0, grad, hessian, max_iter=1_000_000)

        print(u_optim)          