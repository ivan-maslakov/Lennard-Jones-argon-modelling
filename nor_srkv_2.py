import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pygame
import seaborn as sns
from pygame.draw import *

codes = []
for i in [-1, 1, 0]:
    for j in [-1, 1, 0]:
        for k in [-1, 1, 0]:
            codes.append(np.array([i, j, k]))
codes.pop(26)


def rand_direction():
    a = randint(-100, 100)
    c = randint(-100, 100)
    b = randint(-100, 100)
    mdl = (a ** 2 + b ** 2 + c ** 2) ** 0.5
    return np.array([a, b, c]) / mdl


class clone:
    def __init__(self, par, code, x0, x1, x2):
        self.x0 = par.x0 + 2 * lenth * code
        self.x1 = par.x0 + 2 * lenth * code

        self.code = code


class atom:
    def __init__(self, i, j, k, x0, x1, x2, acc, clones):
        l = 2 * lenth / n_1_3
        x = l * i - lenth + l / 2 + randint(-10, 0) / 25 * l
        y = l * j - lenth + l / 2 + randint(-10, 0) / 25 * l
        z = l * k - lenth + l / 2 + randint(-10, 0) / 25 * l
        self.x0 = np.array([x, y, z])
        self.x1 = self.x0 + rand_direction() * v_0 * dt

        self.acc = np.array([0, 0, 0])
        self.clones = []
        for cd in codes:
            self.clones.append(clone(self, cd, 0, 0, 0))


    def teleport(self, atoms):

        if np.dot(self.x0, self.x0) > lenth ** 2:

            if abs(self.x0[1]) >= lenth:

                if self.x0[1] >= lenth and self in atoms:

                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atoms.append(atom(0, 0, 0, 0, 0, 0, 0, []))
                    atom_new = atoms[len(atoms) - 1]
                    atom_new.x0 = atom0x0 - 2 * lenth * np.array([0, 1, 0])
                    atom_new.x1 = atom0x1 - 2 * lenth * np.array([0, 1, 0])

                    atom_new.move_my_clones()
                if self.x0[1] <= -lenth and self in atoms:

                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atoms.append(atom(0, 0, 0, 0, 0, 0, 0, []))
                    atom_new = atoms[len(atoms) - 1]
                    atom_new.x0 = atom0x0 + 2 * lenth * np.array([0, 1, 0])
                    atom_new.x1 = atom0x1 + 2 * lenth * np.array([0, 1, 0])

                    atom_new.move_my_clones()
            if abs(self.x0[0]) >= lenth:
                if self.x0[0] >= lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atoms.append(atom(0, 0, 0, 0, 0, 0, 0, []))
                    atom_new = atoms[len(atoms) - 1]
                    atom_new.x0 = atom0x0 - 2 * lenth * np.array([1, 0, 0])
                    atom_new.x1 = atom0x1 - 2 * lenth * np.array([1, 0, 0])

                    atom_new.move_my_clones()
                if self.x0[0] <= -lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atoms.append(atom(0, 0, 0, 0, 0, 0, 0, []))
                    atom_new = atoms[len(atoms) - 1]
                    atom_new.x0 = atom0x0 + 2 * lenth * np.array([1, 0, 0])
                    atom_new.x1 = atom0x1 + 2 * lenth * np.array([1, 0, 0])

                    atom_new.move_my_clones()
            if abs(self.x0[2]) >= lenth:
                if self.x0[2] >= lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atoms.append(atom(0, 0, 0, 0, 0, 0, 0, []))
                    atom_new = atoms[len(atoms) - 1]
                    atom_new.x0 = atom0x0 - 2 * lenth * np.array([0, 0, 1])
                    atom_new.x1 = atom0x1 - 2 * lenth * np.array([0, 0, 1])

                    atom_new.move_my_clones()
                if self.x0[2] <= -lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atoms.append(atom(0, 0, 0, 0, 0, 0, 0, []))
                    atom_new = atoms[len(atoms) - 1]
                    atom_new.x0 = atom0x0 + 2 * lenth * np.array([0, 0, 1])
                    atom_new.x1 = atom0x1 + 2 * lenth * np.array([0, 0, 1])

                    atom_new.move_my_clones()

    def move_myself(self):
        self.x1, self.x0 = 2 * self.x1 - self.x0 + self.acc * (dt ** 2), self.x1

    def move_my_clones(self):
        for cl in self.clones:
            cl.x0 = self.x0 + 2 * lenth * cl.code
            cl.x1 = self.x1 + 2 * lenth * cl.code


    def accel(self, obj):
        obj_acc = np.array([0, 0, 0])
        vec = obj.x1 - self.x1
        vv = np.dot(vec, vec) / sigma ** 2
        if vv < 6.25:
            obj_acc = (6 / vv ** 3.5 - 12 / vv ** 6.5) * vec / vv ** 0.5
        for cl in obj.clones:
            vec = cl.x1 - self.x1
            vv = np.dot(vec, vec) / sigma ** 2
            if vv < 6.25:
                obj_acc = obj_acc + (-6 / vv ** 3.5 + 12 / vv ** 6.5) * vec / vv ** 0.5

        self.acc = self.acc + 4 * eps / sigma ** 2 / m * obj_acc / 2
        obj.acc = obj.acc - 4 * eps / sigma ** 2 / m * obj_acc / 2

    def e_p(self, obj):
        obj_acc = 0
        vec = obj.x1 - self.x1
        vv = np.dot(vec, vec) / sigma ** 2
        if vv < 6.25:
            obj_acc = -2 * eps * (1 / vv ** 3 - 1 / vv ** 6)
        for cl in obj.clones:
            vec = cl.x1 - self.x1
            vv = np.dot(vec, vec) / sigma ** 2
            if vv < 6.25:
                obj_acc = obj_acc - 2 * eps * (1 / vv ** 3 - 1 / vv ** 6)
        return obj_acc


pygame.init()
FPS = 10000
sc_s = 100
screen = pygame.display.set_mode((6 * sc_s, 6 * sc_s))
screen.fill((255, 255, 255))

sigma = 0.3542 * 10 ** -9
eps = 120 * 1.38 * 10 ** (-23)
k_b = 1.38 * 10 ** (-23)
m = 0.04 / 6 / 10 ** 23


dt = 1 * 10 ** -16
ds = 0.0000001 * sigma

T = 300
v_0 = (3 * k_b * T / m) ** 0.5

n_1_3 = 3
P = 300 * 10 ** 5

lenth = n_1_3 * (8.31 * T / P / 6 / 10 ** 23) ** (1 / 3)

dx_0 = v_0 * dt





t = 0
atoms = []
arr_p = []
arr_t = []
arr_e = []
arr_ek = []
arr_ep = []
clock = pygame.time.Clock()
finished = False



def main(tp):
    sigma = 0.3542 * 10 ** -9
    eps = 120 * 1.38 * 10 ** (-23)
    k_b = 1.38 * 10 ** (-23)
    m = 0.04 / 6 / 10 ** 23

    dt = 1 * 10 ** -15.5
    ds = 0.0000001 * sigma

    T = 300
    v_0 = (3 * k_b * T / m) ** 0.5

    n_1_3 = 2
    P = 300 * 10 ** 5

    lenth = n_1_3 * (8.31 * T / P / 6 / 10 ** 23) ** (1 / 3)
    t = 0
    atoms = []
    arr_p = []
    arr_t = []
    arr_e = []
    arr_ek = []
    arr_ep = []
    p_priv = np.array([0, 0, 0])
    for i in range(n_1_3):
        for j in range(n_1_3):
            for k in range(n_1_3):
                atoms.append(atom(i, j, k, 0, 0, 0, 0, []))
    dp = np.array([0, 0, 0])
    for a in atoms:
        dp = dp + a.x1 - a.x0
    dp = dp / len(atoms)
    for a in atoms:
        a.x1 = a.x1 - dp
    #v_0 = 0
    vsss = []
    for atom0 in atoms:
        vsss.append(np.dot(atom0.x1 - atom0.x0, atom0.x1 - atom0.x0) ** 0.5 / dt)

    #lers = []

    ep0 = 0
    for atom0 in atoms:
        for en_atom in atoms:
            if en_atom != atom0:
                ep0 = ep0 + atom0.e_p(en_atom)
    #screen.fill((255, 255, 255))
    arr_srkv = []
    arr_t_for_srkv = []
    while t < tp:
        if int(t / dt) in np.arange(0, 10 ** 5, 1000):
            vs = vsss
            vs = np.array(vs)
            vmid = ((vs * vs).sum() / len(vs)) ** 0.5
            vmid = vs.sum() / len(vs)
            ed = vs / vs

            srkv = (((vs - vmid * ed) * (vs - vmid * ed)).sum() / len(vs)) ** 0.5

            arr_srkv.append(srkv)
            arr_t_for_srkv.append(t)
            print(srkv)
            print(t)
        '''
        if max(vsss) > 10000:
            break
        '''
        #print(t)
        '''
        lera = 10000
        for a in atoms:
            for ea in atoms:
                fg = np.dot(a.x1 - ea.x1, a.x1 - ea.x1) ** 0.5
                if lera > fg and ea != a:
                    lera = fg
        lers.append(lera)
        '''

        vsss = []
        '''
        ssum = 0
        for atom0 in atoms:
            ssum += np.dot(atom0.x1 - atom0.x0, atom0.x1 - atom0.x0) ** 0.5 / dt
            weewr = atom0.acc
        p_sr = ssum / len(atoms)
        '''
        for atom0 in atoms:
            atom0.acc = np.array([0, 0, 0])

        #arr_p.append((np.dot(p_priv, p_priv)) ** 0.5 / p_sr)

        #p_priv = np.array([0, 0, 0])
        #e_p = 0
        #e_k = 0
        arr_t.append(t)
        # clock.tick(FPS)
        # screen.fill((255, 255, 255))
        t += dt

        #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
                #finished = True

        for atom0 in atoms:
            vsss.append(np.dot(atom0.x1 - atom0.x0, atom0.x1 - atom0.x0) ** 0.5 / dt)
            #for cl in atom0.clones:
                #circle(screen, (0, int(abs(254 * ((cl.x1[2] + lenth) / 2 / lenth))) % 255, 0),
                       #(3 * sc_s + cl.x1[0] / lenth * sc_s, 3 * sc_s + cl.x1[1] / lenth * sc_s), 7)
            #circle(screen, (255, int(abs(254 * ((atom0.x1[2] + lenth) / 2 / lenth))) % 255, 0),
                   #(3 * sc_s + atom0.x1[0] / lenth * sc_s, 3 * sc_s + atom0.x1[1] / lenth * sc_s), 7)

            for en_atom in atoms:
                if en_atom != atom0:
                    atom0.accel(en_atom)
                    #e_p = e_p + atom0.e_p(en_atom)

        for atom0 in atoms:
            atom0.move_myself()
            atom0.teleport(atoms)
            atom0.move_my_clones()
            #v = (atom0.x1 - atom0.x0) / dt
            #e_k += m * np.dot(v, v) / 2
            #p_priv = p_priv + v


        #arr_e.append(e_p - ep0 + e_k)
        #arr_ek.append(e_k)
        #arr_ep.append(e_p)
        # print('E0', e_p + e_k)
        # print('ep', e_p)
        # print('ek', e_k)
        pygame.display.update()
        #vm = max(vsss)
        # dt = ds / vm
    return(arr_srkv, arr_t_for_srkv)
arr_arrvs = []
sch = 0
tp = 1 * 10 ** -14.2
arr_srkv = []
arr_t_for_srkv = []
skv,ttt = main(10 ** -10)
print(skv)
print(ttt)
'''
while tp < 5 * 10 ** 12:
    sch = 0
    vs = []
    while sch < 10:
        vss = main(tp)
        print(sch)
        if max(vss) < 10000:
            for v_ in vss:
                vs.append(v_)

            sch += 1
    vs = np.array(vs)
    vmid = ((vs * vs).sum() / len(vs)) ** 0.5
    ed = vs / vs

    srkv = (((vs - vmid * ed) * (vs - vmid * ed)).sum() / len(vs)) ** 0.5
    arr_srkv.append(srkv)
    arr_t_for_srkv.append(tp)
    print(srkv)
    print(tp)
    tp = tp * 10 ** 0.2
'''
#print()
#print(arr_srkv)
#print(arr_t_for_srkv)
'''
masv = []
for a in arr_arrvs:
    for v in a:
        masv.append(v)
print(masv)
print(len())


a = masv
b = []
for i in a:
    if i < 1000:
        b.append(i)
a = np.array(a)
vsr = (a.sum()) / len(a)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()

print(len(b))
ar = ax.hist(b, 20)

v = np.array(ar[1][0:-1]) * np.array(ar[1][0:-1])
n = np.array(ar[0])
print(len(v), len(n))
print(n)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()
sns.distplot(b, hist=True, kde=True,
             bins=20, color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
fig1 = plt.figure(figsize=(6, 4))
ax1 = fig1.add_subplot()
plt.scatter(v, np.log(n), color='orange', label = 'ek')
v = np.array(ar[1][0:-1])
y = (m/2 / np.pi / k_b / T) ** 0.5 * v * v * np.exp(-1 * m * v * v / 2 / k_b / T)
#plt.scatter(v, y, color='orange', label = 'ek')

ax.grid()
print(ar)
plt.show()
'''
'''
arr_p = np.array(arr_p)
arr_t = np.array(arr_t)
lers = np.array(lers)

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
plt.scatter(arr_t, arr_e, color='b', label = 'e0')
plt.grid()
plt.legend()

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
plt.scatter(arr_t, arr_ep, color='r', label = 'ep')
plt.grid()
plt.legend()


fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
plt.scatter(arr_t, lers / sigma, color='pink', label = 'r')
plt.grid()
plt.legend()

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
plt.scatter(arr_t, arr_ek, color='g', label = 'ek')
plt.grid()
plt.legend()

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
plt.scatter(arr_t, arr_ek, color='g', label = 'ek')
plt.scatter(arr_t, arr_ep, color='r', label = 'ep')
plt.scatter(arr_t, arr_e, color='b', label = 'e0')
plt.grid()
plt.legend()
plt.show()
'''