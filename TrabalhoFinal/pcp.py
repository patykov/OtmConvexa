import numpy as np


def shrink(M, tau):
    return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))


def my_pcp(M, f_shape, delta=1e-3, maxiter=50, connected=True, verbose=True):
    shape = M.shape
    m = np.max(shape)

    # Initialize the tuning parameters.
    lamb = 1.0 / np.sqrt(m)
    neigh = 0.05*np.array(f_shape)
    mu = 0.25 * np.prod(shape) / np.sum(np.abs(M))
    print("mu = {0}".format(mu))
    lamb2 = 10**(-4)

    # Convergence criterion.
    norm = np.sum(M ** 2)

    # Iterate.
    rank = np.min(shape)

    S = np.zeros(shape)
    Y = np.zeros(shape)
    for k in range(maxiter):
        # Update L
        u, s, v = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        s = shrink(s, 1./mu)
        rank = np.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = np.dot(u, np.dot(np.diag(s), v))

        # Update S
        if connected:
            F = []
            # Iterate over lines of s (frames)
            for l, fl in enumerate(S):
                f = np.reshape(fl, f_shape)
                new_f = np.zeros(f_shape)

                for i in range(f_shape[0]):
                    x_min = max(0, int(i-neigh[0]/2))
                    x_max = min(f_shape[0], int(i+neigh[0]/2))
                    for j in range(f_shape[1]):
                        y_min = max(0, int(j-neigh[1]/2))
                        y_max = min(f_shape[1], int(j+neigh[1]/2))

                        block = f[x_min:x_max, y_min:y_max]

                        diff = f[i, j] - block

                        norm2_block = np.linalg.norm(diff, ord=2)
                        prod_fij = - lamb2 * diff/(norm2_block if norm2_block > 0 else 1)

                        # new_f[x_min:x_max, y_min:y_max] += lamb2 * block/(
                        #     norm2_block if norm2_block > 0 else 1)

                        # w = np.exp(-(diff_m**2)/sigma)
                        # prod_fij = w * lamb2 * np.sign(diff_f)

                        new_f[x_min:x_max, y_min:y_max] += prod_fij
                        new_f[i, j] -= sum(sum(prod_fij))

                F.append(np.reshape(new_f, m))

            S_ = M - L + Y/mu - F

        else:
            S_ = M - L + Y/mu

        S = shrink(S_, lamb / mu)

        # Update Y
        step = M - L - S
        Y += mu * step

        # Check for convergence
        err = np.sqrt(abs(np.sum(step ** 2) / norm))
        if verbose:
            print("Iteration {0}: error={1:.3e}, rank={2:d}, nnz={3:d}".format(
                k, err, np.sum(s > 0), np.sum(S > 0)))
        if err < delta:
            return L, S

    print("Convergence not reached in pcp!")
    return L, S
