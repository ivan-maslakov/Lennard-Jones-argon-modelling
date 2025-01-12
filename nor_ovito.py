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
znak = 10

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

n_1_3 = 2
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
for i in range(n_1_3):
        for j in range(n_1_3):
            for k in range(n_1_3):
                atoms.append(atom(i, j, k, 0, 0, 0, 0, []))

coords = []
cocods = []

for at in atoms:
    coords.append([at.x0[0], at.x0[1], at.x0[2]])
coords.append([[0, 0, 0.03 / lenth],[],[]])
coords = np.array(coords) / 0.03 / lenth
with open('atoms.xyz', 'w') as f:
    f.write(f'{len(coords)}\n')
    #f.write(f'Lattice="{0.03 / lenth} 0.0 0.0 0.0 {0.03 / lenth} 0.0 0.0 0.0 {0.03 / lenth}" Properties=species:S:1:pos:R:3 ')
    f.write(f'La\n')
    #f.write(f'{len(coords)} atoms\n\n') # количество атомов
    #f.write('1 atom types\n\n')
    #f.write('0.0 1.0 xlo xhi\n') # границы ящика
    #f.write('0.0 1.0 ylo yhi\n')
    #f.write('0.0 1.0 zlo zhi\n\n')
    #f.write('Atoms\n\n')
    for i, coord in enumerate(coords):
        if i != len(coords) - 1:
            f.write(f'Ar {coord[0]} {coord[1]} {coord[2]}\n')
        else:
            f.write(f'Ar {coord[0]} {coord[1]} {coord[2]}\n')
with open('atoms.xyz', 'w') as f:
    f.write(f'{len(coords)}\n')
    f.write(f'La\n')
    #f.write(f'Lattice="{0.03 / lenth} 0.0 0.0 0.0 {0.03 / lenth} 0.0 0.0 0.0 {0.03 / lenth}" Properties=species:S:1:pos:R:3 ')
    #f.write(f'{len(coords)} atoms\n\n') # количество атомов
    #f.write('1 atom types\n\n')
    #f.write('0.0 1.0 xlo xhi\n') # границы ящика
    #f.write('0.0 1.0 ylo yhi\n')
    #f.write('0.0 1.0 zlo zhi\n\n')
    #f.write('Atoms\n\n')
    for i, coord in enumerate(coords):
        if i != len(coords) - 1:
            f.write(f'Ar {coord[0]} {coord[1]} {coord[2]}\n')
        else:
            f.write(f'Ar {coord[0]} {coord[1]} {coord[2]}\n')
def main():
    sigma = 0.3542 * 10 ** -9
    eps = 120 * 1.38 * 10 ** (-23)
    k_b = 1.38 * 10 ** (-23)
    m = 0.04 / 6 / 10 ** 23


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


    while t < 1 * 10 ** -13:

        if max(vsss) > 10000:
            print(t)
            break

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
        coords = []

        for at in atoms:
            coords.append([at.x0[0], at.x0[1], at.x0[2]])
        coords = np.array(coords) / 0.03 / lenth
        if int(t / dt) % 10 == 0:
            with open('atoms.xyz', 'a+') as f:
                f.write(f'{len(coords)}\n')
                #f.write(f'Lattice="{0.03 / lenth} 0.0 0.0 0.0 {0.03 / lenth} 0.0 0.0 0.0 {0.03 / lenth}" Properties=species:S:1:pos:R:3 ')
                f.write(f'La\n')
                # f.write(f'{len(coords)} atoms\n\n') # количество атомов
                # f.write('1 atom types\n\n')
                # f.write('0.0 1.0 xlo xhi\n') # границы ящика
                # f.write('0.0 1.0 ylo yhi\n')
                # f.write('0.0 1.0 zlo zhi\n\n')
                # f.write('Atoms\n\n')
                for i, coord in enumerate(coords):
                    f.write(f'Ar {coord[0]} {coord[1]} {coord[2]}\n')

    return(vsss)
main()