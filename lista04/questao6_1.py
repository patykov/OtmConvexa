import numpy as np


def get_filter_parameters():
    w_p = 0.45 * np.pi
    w_a = 0.5 * np.pi
    delta_p = 0.025
    delta_a = 10**(-2)
    gamma = 3 * 10**3
    Mp = 80
    Ma = 10

    # Calculating Q_l:
    N = 84
    s = ((N + 2) / 2)
    Q_l1 = np.zeros((s, s))
    Q_l2 = np.zeros((s, s))
    for i in range(1, s):
        for j in range(1, s):
            if i == j:
                Q_l1[i, j] = w_p / 2 + np.sin(2 * (i - 1) * w_p) / (4 * (i - 1))
                Q_l2[i, j] = gamma * ((np.pi - w_a) / 2 - np.sin(2 * (i - 1) * w_a) / (4 * (i - 1)))
            else:
                Q_l1[i, j] = np.sin((i - j) * w_p) / (2 * (i - j)) + np.sin(
                    (i + j - 2) * w_p) / (2 * (i + j - 2))
                Q_l2[i, j] = (-gamma / 2) * (np.sin((i - j) * w_a) / (i - j) + np.sin(
                    (i + j - 2) * w_a) / (i + j - 2))

    Q_l1[0, 0] = w_p
    Q_l2[0, 0] = gamma * (np.pi - w_a)
    Q_l = Q_l1 + Q_l2

    # Calculating b_l:
    b_l = np.zeros((s))
    for i in range(1, s):
        b_l[i] = np.sin((i - 1) * w_p) / (i - 1)
    b_l[0] = w_p

    # Calculating bp:
    bp = np.concatenate(((1 + delta_p) * np.ones((Mp, 1)), (-1 + delta_p) * np.ones((Mp, 1))),
                        axis=0)

    # Calculating ba:
    ba = (delta_a) * np.ones((2 * Ma, 1))

    # Calculating Ap:
    Ap_up = []
    Sp = np.linspace(0, w_p, Mp)
    for e in range(1, Mp):
        Ap_up.append([np.cos(d * Sp[e]) for d in range(N / 2 + 1)])
    Ap_down = -1 * np.array(Ap_up)
    Ap = np.concatenate((Ap_up, Ap_down), axis=0)

    # Calculating Aa:
    Aa_up = []
    Sa = np.linspace(w_a, np.pi, Ma)
    for f in range(1, Ma):
        Aa_up.append([np.cos(d * Sa[f]) for d in range(N / 2 + 1)])
    Aa_down = -1 * np.array(Aa_up)
    Aa = np.concatenate((Aa_up, Aa_down), axis=0)

    return np.array(Q_l), np.array(b_l), np.array(Ap), bp, np.array(Aa), np.array(ba)


if __name__ == '__main__':
    Q_l, b_l, Ap, bp, Aa, ba = get_filter_parameters()
    print('Ql', Q_l.shape)
    print('bl', b_l.shape)
    print('Ap', Ap.shape)
    print('bp', bp.shape)
    print('Aa', Aa.shape)
    print('ba', ba.shape)
