import numpy as np

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=-1).reshape(shape)
    return quat

def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    
    xyz = a[:, :3]
    t = np.cross(xyz, b) * 2
    return (b + a[:, 3:] * t + np.cross(xyz, t)).reshape(shape)

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return np.concatenate((-a[:, :3], a[:, -1:]), axis=-1).reshape(shape)

def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)

def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1

def tf_inverse2(tf):
    q, t = tf[3:7], tf[:3]
    q_inv, t_inv = tf_inverse(q, t)
    return np.concatenate([t_inv, q_inv])

def tf_combine2(tf1, tf2):
    q1, t1 = tf1[3:7], tf1[:3]
    q2, t2 = tf2[3:7], tf2[:3]
    q_combine, t_combine = tf_combine(q1, t1, q2, t2)
    return np.concatenate([t_combine, q_combine])