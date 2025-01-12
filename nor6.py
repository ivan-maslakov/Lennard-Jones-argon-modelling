import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pygame
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
        ll =1.03* l
        x = ll * i - lenth + l / 2
        y = l * j - lenth + l / 2
        z = l * k - lenth + l / 2
        z = 0
        self.x0 = np.array([x, y, z])
        self.x1 = self.x0 + rand_direction() * v_0 * dt / 1

        self.acc = np.array([0, 0, 0])
        self.clones = []
        for cd in codes:
            self.clones.append(clone(self, cd, 0, 0, 0))
        self.clones = []

    def teleport(self):
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
        if vv < 200:
            obj_acc = 4 * eps / sigma ** 2 / m * (6 / vv ** 3.5 - 12 / vv ** 6.5) * vec / vv ** 0.5
        for cl in obj.clones:
            vec = cl.x1 - self.x1
            vv = np.dot(vec, vec) / sigma ** 2
            if vv < 200:
                obj_acc = obj_acc + 4 * eps / sigma ** 2 / m * (-6 / vv ** 3.5 + 12 / vv ** 6.5) * vec / vv ** 0.5
        self.acc = self.acc + obj_acc / 2
        obj.acc = obj.acc - obj_acc / 2

    def e_p(self, obj):
        obj_acc = 0
        vec = obj.x1 - self.x1
        vv = np.dot(vec, vec) / sigma ** 2
        if vv < 200:
            obj_acc = -2 * eps * (1 / vv ** 3 - 1 / vv ** 6)
        for cl in obj.clones:
            vec = cl.x1 - self.x1
            vv = np.dot(vec, vec) / sigma ** 2
            if vv < 200:
                obj_acc = obj_acc - 2 * eps * (1 / vv ** 3 - 1 / vv ** 6)
        return obj_acc


pygame.init()
FPS = 10000
sc_s = 300
screen = pygame.display.set_mode((2 * sc_s, 2 * sc_s))
screen.fill((255, 255, 255))

sigma = 0.3542 * 10 ** -9
eps = 120 * 1.38 * 10 ** (-23)
k_b = 1.38 * 10 ** (-23)
m = 0.04 / 6 / 10 ** 23
dt = 1 * 10 ** -16
T = 300
v_0 = (3 * k_b * T / m) ** 0.5
v_0 = 0
n_1_3 = 2
P = 900 * 10 ** 5
lenth = 0.9 * sigma
lenth = (n_1_3 * 8.31 * T / P / 6 / 10 ** 23) ** (1 / 3)
lenth = 0.85 * sigma
print('lenth', lenth / sigma)
dx_0 = v_0 * dt
print('sdfsavav',v_0)



p_priv = np.array([0, 0, 0])
t = 0
atoms = []
arr_p = []
arr_t = []
arr_e = []
arr_ek = []
arr_ep = []
clock = pygame.time.Clock()
finished = False

for i in range(n_1_3):
    for j in range(1):
        for k in range(1):
            atoms.append(atom(i, j, k, 0, 0, 0, 0, []))
dp = np.array([0, 0, 0])
for a in atoms:
    dp = dp + a.x1 - a.x0
dp = dp / len(atoms)
for a in atoms:
    a.x1 = a.x1 - dp
v_0 = 0
vsss = []
for atom0 in atoms:
    vsss.append(np.dot(atom0.x1 - atom0.x0, atom0.x1 - atom0.x0) ** 0.5 / dt)
print(vsss)
lers = []


ep0 = 0
for atom0 in atoms:
    for en_atom in atoms:
        if en_atom != atom0:
            ep0 = ep0 + atom0.e_p(en_atom)


ep10 = atoms[0].e_p(atoms[1])
arr_ep1 = []
while not finished:
    ep10 = 0

    lera = 10000
    for a in atoms:
        for ea in atoms:
            if ea != a:
                fg = np.dot(a.x0 - ea.x0,a.x0 - ea.x0) ** 0.5
                if lera > fg:
                    lera = fg
    lers.append(lera / sigma)

    print(t)
    print(t)
    print(t)
    vsss = []
    print(vsss)

    ssum = 0
    for atom0 in atoms:
        ssum += np.dot(atom0.x1 - atom0.x0, atom0.x1 - atom0.x0) ** 0.5 / dt
        weewr = atom0.acc
    p_sr = ssum / len(atoms)

    for atom0 in atoms:
        atom0.acc = np.array([0,0,0])

    arr_p.append((np.dot(p_priv, p_priv)) ** 0.5 / p_sr)

    p_priv = np.array([0, 0, 0])
    e_p = 0
    e_k = 0
    arr_t.append(t)
    clock.tick(FPS)
    screen.fill((255, 255, 255))
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            finished = True

    for atom0 in atoms:
        vsss.append(np.dot(atom0.x1 - atom0.x0, atom0.x1 - atom0.x0) ** 0.5 / dt)
        circle(screen, (255, int(abs(254 * ((atom0.x1[2] + lenth) / 2 / lenth))) % 255, 0),
               (sc_s + atom0.x1[0] / lenth * sc_s, sc_s + atom0.x1[1] / lenth * sc_s), 7)

        for en_atom in atoms:
            if en_atom != atom0:
                atom0.accel(en_atom)
                e_p = e_p + atom0.e_p(en_atom)
    ep1 = atoms[0].e_p(atoms[1])
    for atom0 in atoms:
        atom0.move_myself()
        atom0.move_my_clones()
        v = (atom0.x1 - atom0.x0) / dt
        e_k += m * np.dot(v, v) / 2
        p_priv = p_priv + v
        #atom0.teleport()
    if 1.5 > t > 1 * 10 ** -7:
        print('adfadfa', -1 * e_k / e_p)
    arr_e.append(e_p - ep0 + e_k)
    arr_ek.append(e_k)
    arr_ep.append(e_p - ep0)
    arr_ep1.append(ep1 - ep10)
    print('E0', e_p - ep0 + e_k)
    print('ep', e_p - ep0)
    print('ek', e_k)
    pygame.display.update()

print(vsss)
arr_p = np.array(arr_p)
arr_t = np.array(arr_t)
arr_ep = np.array(arr_ep)
arr_ek = np.array(arr_ek)
'''
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot()
plt.scatter(arr_t, arr_ep + arr_ek, color='r', label = 'e0')
plt.grid()
plt.legend()
'''
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot()
plt.scatter(arr_t, lers, color='g', label = 'r')
plt.grid()
plt.legend()

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot()
plt.scatter(arr_t, arr_ep, color='b', label = 'ep')
plt.scatter(arr_t, arr_ek, color='orange', label = 'ek')
plt.scatter(arr_t, arr_ep + arr_ek, color='r', label = 'ep')
plt.grid()
plt.legend()

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot()
plt.scatter(lers, arr_ek, color='orange', label = 'ek')
plt.grid()
plt.legend()


fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot()
plt.scatter(lers, arr_ep + ep0, color='pink', label = 'ep')
plt.grid()
plt.legend()
plt.show()