from sklearn.neighbors import KDTree

import numpy as np
#
# class MyNode:
#     def __init__(self, point = []):
#         self.point = point
#         self.isLeaf = True
#
#     def set_v_right(self, point):
#         self.isLeaf = False
#         self.v_right = point
#     def set_v_left(self, point):
#         self.isLeaf = False
#         self.v_left = point
#
#
# def build_kd_tree(points=[], depth=0):
#     if len(points) == 0:
#         return
#     if len(points) == 1:
#         return MyNode(points[0])
#     else:
#         if depth%2 == 0:
#             ax = 0
#         else:
#             ax = 1
#         if len(points)%2==0:
#             points.append([0,0])
#         md = np.median([el[ax] for el in points])
#
#         left_p = []
#         i = 0
#         while points[i][ax] != md:
#             left_p.append(points[i])
#             i += 1
#         v = MyNode(points[i])
#         i += 1
#         right_p = []
#         while i < len(points):
#             right_p.append(points[i])
#             i += 1
#         if len(left_p):
#             v_left = build_kd_tree(points=left_p, depth=depth + 1)
#             v.set_v_left(v_left)
#
#         if len(right_p):
#             v_right = build_kd_tree(points=right_p, depth=depth + 1)
#             v.set_v_right(v_right)
#         return v
#

def main():
    n = 99
    #points = [[np.random.rand(), np.random.rand(),np.random.rand()] for i in range(n)]
    points = np.random.random((10, 3))
    kdt = KDTree(points, leaf_size=30, metric='euclidean')
    print(points[0])
    dist, ind =kdt.query([points[0]],k=2)
    print(points[ind])
#     #k = 3
#     points = [[np.random.rand(),np.random.rand()]for i in range(n)]
#     # besser, wenn ich das rekursiv aufrufe->append
#
#     #print(testvec[0])
#     #mn1 = MyNode(testvec[0])
#     #mn1.set_v_left(MyNode(testvec[1]))
#     #points = np.random.rand(n,k)
#     # warum bekomme ich hier nicht den richtigen median zurück?
#     # bei gerade anzahl wird zwischen den mittleren elementen interpoliert.
#     ## je nach durchlauf an ax = i feststellen
#     kd =build_kd_tree(points=points,depth=0)
#
if __name__ == '__main__':
    main()