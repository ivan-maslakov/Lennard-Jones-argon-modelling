import numpy as np
import matplotlib.pyplot as plt
from random import randint
import numpy.linalg as linalg
import pygame
import seaborn as sns
from pygame.draw import *
from numba import jit
from numba.typed import List
from numba.experimental import jitclass
from numba import int64, float64, deferred_type
from numba.typed import List
import numba as nb





codes = []
for i in [-1, 1, 0]:
    for j in [-1, 1, 0]:
        for k in [-1, 1, 0]:
            codes.append(np.array([i, j, k]))
codes.pop(26)


@jit(nopython= True)
def nc_acc(x1, x1_, clones_x1_):
    obj_acc = np.array([float(0), float(0), float(0)])
    vec = x1_ - x1
    vv = (vec * vec).sum() / sigma ** 2

    if vv < 6.25 and vv != 0:
        obj_acc = (6 / vv ** 3.5 - 12 / vv ** 6.5) * vec / vv ** 0.5
    for cx1_ in clones_x1_:
        vec = cx1_ - x1
        vv = (vec * vec).sum() / sigma ** 2
        if vv < 6.25 and vv != 0:
            obj_acc = obj_acc + (-6 / vv ** 3.5 + 12 / vv ** 6.5) * vec / vv ** 0.5

    return 4 * eps / sigma ** 2 / m * obj_acc


def nc_acc_new(x1, clones_x1):
    vec = clones_x1 - np.array([x1 for _ in range(27)])
    b = linalg.norm(vec, axis=1) / sigma ** 2 #vv
    c = (vec * (6 / b ** 4 - 12 / b ** 7)[:, None]).sum(axis=0)
    return 4 * eps / sigma ** 2 / m * c

@jit(nopython= True)
def nc_e_p(x1, x1_, clones_x1_):
    obj_acc = 0
    vec = x1_ - x1
    vv = (vec * vec).sum() / sigma ** 2
    if vv < 6.25 and vv !=0:
        obj_acc = -2 * eps * (1 / vv ** 3 - 1 / vv ** 6)
    for cx1_ in clones_x1_:
        vec = cx1_ - x1
        vv = (vec * vec).sum() / sigma ** 2
        if vv < 6.25 and vv != 0:
            obj_acc = obj_acc - 2 * eps * (1 / vv ** 3 - 1 / vv ** 6)
    return obj_acc


@jit(nopython= True)
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
        x01 = [[x, y, z]]
        npx01 = np.array([x, y, z])
        for cd in codes:
            x01.append(npx01 + 2 * lenth * cd)
        self.x0 = np.array(x01)
        self.x1 = self.x0 + rand_direction() * v_0 * dt

        self.acc = np.array([0, 0, 0])
        #self.clones = []
        #for cd in codes:
            #self.clones.append(clone(self, cd, 0, 0, 0))

    def accel(self, obj):
        vec = obj.x0 - np.array([self.x0[0] for _ in range(27)])
        bbb = linalg.norm(vec, axis=1) / sigma
        #ccc = (vec * (6 / bbb ** 4 - 12 / bbb ** 7)[:, None]).sum(axis=0)
        da = 4 * eps / sigma ** 2 / m * ((vec * (6 / bbb ** 8 - 12 / bbb ** 14)[:, None]).sum(axis=0))
        self.acc = self.acc + da
        obj.acc = obj.acc - da



    def teleport(self, atoms):
        u = self.x0[0]
        q = self.x0[1]
        if np.dot(u, u) > lenth ** 2:

            if abs(u[1]) >= lenth:

                if u[1] >= lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atom_new = atom(0, 0, 0, 0, 0, 0, 0, [1])
                    atoms.add(atom_new)
                    atom_new.x0 = atom0x0 - 2 * lenth * np.array([0, 1, 0])
                    atom_new.x1 = atom0x1 - 2 * lenth * np.array([0, 1, 0])

                    #atom_new.move_my_clones()
                if u[1] <= -lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atom_new = atom(0, 0, 0, 0, 0, 0, 0, [1])
                    atoms.add(atom_new)
                    atom_new.x0 = atom0x0 + 2 * lenth * np.array([0, 1, 0])
                    atom_new.x1 = atom0x1 + 2 * lenth * np.array([0, 1, 0])

                    #atom_new.move_my_clones()
            if abs(u[0]) >= lenth:
                if u[0] >= lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atom_new = atom(0, 0, 0, 0, 0, 0, 0, [1])
                    atoms.add(atom_new)
                    atom_new.x0 = atom0x0 - 2 * lenth * np.array([1, 0, 0])
                    atom_new.x1 = atom0x1 - 2 * lenth * np.array([1, 0, 0])

                    #atom_new.move_my_clones()
                if u[0] <= -lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atom_new = atom(0, 0, 0, 0, 0, 0, 0, [1])
                    atoms.add(atom_new)
                    atom_new.x0 = atom0x0 + 2 * lenth * np.array([1, 0, 0])
                    atom_new.x1 = atom0x1 + 2 * lenth * np.array([1, 0, 0])

                    #atom_new.move_my_clones()
            if abs(u[2]) >= lenth:
                if u[2] >= lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atom_new = atom(0, 0, 0, 0, 0, 0, 0, [1])
                    atoms.add(atom_new)
                    atom_new.x0 = atom0x0 - 2 * lenth * np.array([0, 0, 1])
                    atom_new.x1 = atom0x1 - 2 * lenth * np.array([0, 0, 1])

                    #atom_new.move_my_clones()
                if u[2] <= -lenth and self in atoms:
                    atom0x0 = self.x0
                    atom0x1 = self.x1

                    atoms.remove(self)
                    atom_new = atom(0, 0, 0, 0, 0, 0, 0, [1])
                    atoms.add(atom_new)

                    atom_new.x0 = atom0x0 + 2 * lenth * np.array([0, 0, 1])
                    atom_new.x1 = atom0x1 + 2 * lenth * np.array([0, 0, 1])

                    #atom_new.move_my_clones()


    def move_myself(self):
        self.x1, self.x0 = 2 * self.x1 - self.x0 + self.acc * (dt ** 2), self.x1


    def move_my_clones(self):
        for cl in self.clones:
            cl.x0 = self.x0 + 2 * lenth * cl.code
            cl.x1 = self.x1 + 2 * lenth * cl.code





    def e_p(self, obj):
        x1 = self.x1
        x1_ = self.x1
        clones_x1_ = List()
        [clones_x1_.append(x.x1) for x in obj.clones]

        return nc_e_p(x1, x1_, clones_x1_)



#pygame.init()
FPS = 10000
sc_s = 300
#screen = pygame.display.set_mode((2 * sc_s, 2 * sc_s))
#screen.fill((255, 255, 255))

sigma = 0.3542 * 10 ** -9
eps = 120 * 1.38 * 10 ** (-23)
k_b = 1.38 * 10 ** (-23)
m = 0.04 / 6 / 10 ** 23

dt = 10 ** -16
T = 300
v_0 = (3 * k_b * T / m) ** 0.5
#v_0 = 0
n_1_3 = 3
P = 300 * 10 ** 5

lenth = n_1_3 * (8.31 * T / P / 6 / 10 ** 23) ** (1 / 3)

clock = pygame.time.Clock()
finished = False
atoms = set()




with open('a.xyz', 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = [float(x) for x in lines[i].split()]
    for i in range(n_1_3 ** 3):
        l = lines[i]
        x00 = l[0]
        x01 = l[1]
        x02 = l[2]
        x10 = l[3]
        x11 = l[4]
        x12 = l[5]
        a = atom(i, j, k, 0, 0, 0, 0, [1])
        x011 = [[x00, x01, x02]]
        npx01 = np.array([x00, x01, x02])
        for cd in codes:
            x011.append(npx01 + 2 * lenth * cd)
        a.x0 = np.array(x011)
        x111 = [[x10, x11, x12]]
        npx11 = np.array([x10, x11, x12])
        for cd in codes:
            x111.append(npx11 + 2 * lenth * cd)
        a.x1 = np.array(x111)
        atoms.add(a)



dp = np.array([0, 0, 0])
for a in atoms:
    dp = dp + a.x1 - a.x0
dp = dp / len(atoms)
for a in atoms:
    a.x1 = a.x1 - dp
# v_0 = 0


def write():
    ans = []
    for a in atoms:
        for c in codes:
            ans.append(2 * lenth * c + a.x0)
    return ans


def main(tp):
    t = 2 * 10 ** -12

    arr_t = []
    arr_e = []
    arr_ek = []
    arr_ep = []
    atoms_prop = []
    vsss = []
    for atom0 in atoms:
        vsss.append(np.dot(atom0.x1[0] - atom0.x0[0], atom0.x1[0] - atom0.x0[0]) ** 0.5 / dt)
    vsssx = []
    vsssy = []
    vsssz = []


    ep0 = 0
    #for atom0 in atoms:
        #for en_atom in atoms:
            #if en_atom != atom0:
                #ep0 = ep0 + atom0.e_p(en_atom)
    #screen.fill((255, 255, 255))


    while t - 2 * 10 ** -12 < tp:
        if (t // dt) % 100 == 0:
            ep = 0
            '''
            for i in range(len(vsss)):
                vsss[i] = np.dot(vsss[i], vsss[i]) ** 0.5
            if max(vsss) > 10000:
                return(arr_ek, arr_t)
                print('EXPLOSION')
                break
            '''
            print(t)
            if (t // dt) % 100 == 0:
                arr_t.append(t)
                e_k = 0

                for atom0 in atoms:
                    vx = (atom0.x1[0][0] - atom0.x0[0][0]) / dt
                    vy = (atom0.x1[0][1] - atom0.x0[0][1]) / dt
                    vz = (atom0.x1[0][2] - atom0.x0[0][2]) / dt
                    v = (atom0.x1[0] - atom0.x0[0]) / dt
                    vsss.append(v)
                    vsssx.append(vx)
                    vsssy.append(vy)
                    vsssz.append(vz)
                    e_k += np.dot(v, v) * m / 2

                arr_ek.append(e_k)


        t += dt

        #for atom0 in atoms:
            #atom0.acc = np.array([0, 0, 0])




        #e_p = 0


        # clock.tick(FPS)
        #screen.fill((255, 255, 255))


        #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
                #finished = True
        for a in atoms:
            a.acc = np.array([0, 0, 0])
        used = set()
        for atom0 in atoms:

            #vsss.append(np.dot(atom0.x1 - atom0.x0, atom0.x1 - atom0.x0) ** 0.5 / dt)
            #for cl in atom0.clones:
                #circle(screen, (0, int(abs(254 * ((cl.x1[2] + lenth) / 2 / lenth))) % 255, 0),
                       #(3 * sc_s + cl.x1[0] / lenth * sc_s, 3 * sc_s + cl.x1[1] / lenth * sc_s), 7)
            #circle(screen, (255, int(abs(254 * ((atom0.x1[0][2] + lenth) / 2 / lenth))) % 255, 0),
                   #(sc_s + atom0.x1[0][0] / lenth * sc_s, sc_s + atom0.x1[0][1] / lenth * sc_s), 7)

            for en_atom in atoms:
                st = {atom0, en_atom}
                if en_atom != atom0 and st not in used:
                    atom0.accel(en_atom)
                    used.add((atom0, en_atom))
                    used.add((en_atom, atom0))
                    #e_p = e_p + atom0.e_p(en_atom)

        for atom0 in atoms:
            atom0.move_myself()
            atom0.teleport(atoms)
            #atom0.move_my_clones()
            #v = (atom0.x1 - atom0.x0) / dt
            #e_k += m * np.dot(v, v) / 2
            #p_priv = p_priv + v


        #arr_e.append(e_p - ep0 + e_k)
        #arr_ek.append(e_k)
        #arr_ep.append(e_p)
        # print('E0', e_p + e_k)
        # print('ep', e_p)
        # print('ek', e_k)
        #pygame.display.update()
        #vm = max(vsss)
        # dt = ds / vm
    new_at = write()
    return(arr_ek, arr_t, new_at, vsssx, vsssy, vsssz)

tp = 0.1 * 10 ** -12
are_k, art, new_at, vsssx, vsssy, vsssz = main(tp)
'''
atoms = set()
for a in new_at:
    at = atom(1, 1, 1, 0, 0, 0, 0, [1])
    at.x0 = a
    atoms.add(at)
tp = 1 * 10 ** -12
are_k, art, new_at = main(tp)
'''
vsss = []
for v in vsssx:
    vsss.append(v)
for v in vsssy:
    vsss.append(v)
for v in vsssz:
    vsss.append(v)


a = abs(np.array(vsss))
b = []
for i in a:
    if i < 600:
        b.append(float(i))
a = np.array(a)
vsr = (a.sum()) / len(a)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()


ar = ax.hist(b, 20)
for i in range(len(ar[0])):
    ar[0][i] = float(ar[0][i])
for i in range(len(ar[1])):
    ar[1][i] = float(ar[1][i])
v = np.array(ar[1][0:-1]) * np.array(ar[1][0:-1])
n = np.array(ar[0])

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()
sns.distplot(b, hist=True, kde=True,
             bins=20, color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
ax.grid()
fig1 = plt.figure(figsize=(6, 4))
ax1 = fig1.add_subplot()
ax1.scatter(v, np.log(n), color='orange', label = 'ek')
v = list(v)
n = list(n)
for i in range(len(v)):
    if i < len(v) and (v[i] == 0 or n[i] == 0):
        v.pop(i)
        n.pop(i)
v = np.array(v)
n = np.array(n)
print(*v)
print(*n)
d = np.polyfit(v, np.log(n), 1)
kk = d[0]
bb = d[1]
ax1.plot([0, v[len(v) - 1]], [bb, bb + kk * v[len(v) - 1]], color = 'b')
ax1.grid()
v = list(v)
n = list(n)
for i in range(len(v)):
    if v[i] == 0:
        v.pop(i)
        n.pop(i)
print(type(np.array([1]).sum()))


fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot()
#plt.scatter(art[:-21], are_k[:-21], color='b', label = 'ek')

#are_k, art = main(tp)


#plt.scatter(art[:-21], are_k[:-21], color='r', label = 'ek')

#are_k, art = main(tp)
cp = are_k

plt.scatter(art, are_k, color='b', label = 'ek')
#plt.scatter(art[:-21], are_k[:-21], color='b', label = 'ek')

ax2.grid()
ax2.legend()

plt.show()

