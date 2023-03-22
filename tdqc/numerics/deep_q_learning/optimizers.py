"""Module defining optimizers

They are used to calculate the action maximizing Q(s, a) = argmax_a Q(s, a)
"""


import numpy as np


class Optimizer:
    def __init__(self, initialization):
        self.initialization = initialization

    def initialize(self, n_inits, dim):
        if self.initialization == 'random':
            a_inits = np.random.uniform(-1, 1, size=(n_inits, dim))
        elif self.initialization == 'uniform':
            values = np.linspace(-1, 1, num=self.n_inits, endpoint=True)
            a_inits = np.array([np.full(dim, a) for a in values])
        elif self.initialization == 'fixed random':
            if hasattr(self, 'fixed_a_inits'):
                pass
            else:
                self.fixed_a_inits = np.random.uniform(
                    -1, 1, size=(n_inits, dim)
                )
            a_inits = self.fixed_a_inits
        else:
            raise ValueError
        return a_inits


class NAGOptimizer(Optimizer):
    def __init__(self,
                 learning_rate,
                 momentum,
                 convergence_threshold,
                 n_iterations,
                 n_inits,
                 initialization
                 ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.convergence_threshold = convergence_threshold
        self.n_inits = n_inits
        self.n_iterations = n_iterations
        super().__init__(initialization)

    def run(self, dim, evaluate_output, evaluate_gradient):
        """
        maximize output q(a) following the update rule
                v_t = γ * v_{t-1} + η * ∇J(a + γ * v_{t-1})
        γ: momentum, η: learning_rate, J: cost function, v: update vector
        """

        a_inits = self.initialize(n_inits=self.n_inits, dim=dim)
        q_max = -np.infty
        a_max = a_inits[0]
        for i, a in enumerate(a_inits):
            update_vec = 0.0 * a
            for i_iter in range(self.n_iterations):
                update_vec *= self.momentum
                grad = evaluate_gradient(a + update_vec)
                update_vec += self.learning_rate * grad
                a, a_old = a + update_vec, a
                #  if (i_iter > 10 and
                #          np.linalg.norm(a-a_old) < self.convergence_threshold):
                if (np.linalg.norm(a-a_old) < self.convergence_threshold):
                    break

            q = evaluate_output(a)
            if q > q_max:
                a_max, q_max = a, q

        return (a_max, q_max)


class AdamOptimizer(Optimizer):
    def __init__(self,
                 alpha=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 convergence_threshold=0.005,
                 n_iterations=1000,
                 n_inits=1,
                 initialization='random'
                 ):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.convergence_threshold = convergence_threshold
        self.n_inits = n_inits
        self.n_iterations = n_iterations
        super().__init__(initialization)

    def run(self, dim, evaluate_output, evaluate_gradient):
        """
        maximize output q(a) following the update rule
                g_t = ∇J(a)
                m_t = β1*m_{t-1} + (1 - β1)*g_t
                v_t = β2*v_{t-1} + (1 - β2)*g_t^2 (elementwise square)
                mhat_t = m_t / (1 - β1^t) (bias-corrected)
                vhat_t = v_t / (1 - β2^t) (bias-corrected)
                a_{t+1} = a_t + α * mhat_t / (sqrt(vhat_t) + ε)
                (+ sign for gradient ascent)

        α: learning_rate, J: cost function,
        m_t: EMA of gradient (sim to momentum) with coef β1,
        v_t EMA of square of gradient with coef β2
        """
        a_inits = self.initialize(n_inits=self.n_inits, dim=dim)
        q_max = -np.infty
        a_max = a_inits[0]
        for i, a in enumerate(a_inits):
            update_vec = 0.0 * a
            m_t, v_t = 0.0, 0.0
            for t in range(self.n_iterations):
                grad = evaluate_gradient(a)
                grad2 = grad * grad
                m_t = self.beta_1 * m_t + (1. - self.beta_1) * grad
                v_t = self.beta_2 * v_t + (1. - self.beta_2) * grad2
                m_hat = m_t / (1. - self.beta_1 ** (t + 1))
                v_hat = v_t / (1. - self.beta_2 ** (t + 1))
                a_old = a
                # + sign because we are doing gradient `ascent`
                a += self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
                if (t > 10 and
                        np.linalg.norm(a-a_old) < self.convergence_threshold):
                #  if (np.linalg.norm(a-a_old) < self.convergence_threshold):
                    break

            q = evaluate_output(a)
            if q > q_max:
                a_max, q_max = a, q

        return (a_max, q_max)
