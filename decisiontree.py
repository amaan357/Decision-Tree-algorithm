import math
import numpy as np
import pandas as pd
import random
import sys


class Node:
    def __init__(self, key):
        self.key = key
        self.order = None
        self.left = None
        self.right = None

    def isLeaf(self):
        return not (self.right or self.left)



def entropy(p,n):
    if p == n:
        return 1
    a = p/(p+n)
    b = n/(p+n)
    return -a*math.log2(a)-b*math.log2(b)

def gain(x,y):
    E = x['Class'].value_counts()
    a = x.loc[x[y] == 1, 'Class'].value_counts()
    b = x.loc[x[y] == 0, 'Class'].value_counts()
    if len(a) == 1 and len(b) == 1:
        return entropy(E[1], E[0])
    elif len(a) == 1 or a.empty:
        return entropy(E[1], E[0])-((b[1]+b[0])/(E[1]+E[0]))*entropy(b[1], b[0])
    elif len(b) == 1 or b.empty:
        return entropy(E[1], E[0])-((a[1]+a[0])/(E[1]+E[0]))*entropy(a[1], a[0])
    else:
        return entropy(E[1], E[0])-((a[1]+a[0])/(E[1]+E[0]))*entropy(a[1], a[0])-((b[1]+b[0])/(E[1]+E[0]))*entropy(b[1], b[0])

def vimp(p,n):
    if p == n:
        return 0.25
    return p*n / (p + n)**2

def gain2(x,y):
    E = x['Class'].value_counts()
    a = x.loc[x[y] == 1, 'Class'].value_counts()
    b = x.loc[x[y] == 0, 'Class'].value_counts()
    if len(a) == 1 and len(b) == 1:
        return vimp(E[1], E[0])
    elif len(a) == 1 or a.empty:
        return vimp(E[1], E[0])-((b[1]+b[0])/(E[1]+E[0]))*vimp(b[1], b[0])
    elif len(b) == 1 or b.empty:
        return vimp(E[1], E[0])-((a[1]+a[0])/(E[1]+E[0]))*vimp(a[1], a[0])
    else:
        return vimp(E[1], E[0])-((a[1]+a[0])/(E[1]+E[0]))*vimp(a[1], a[0])-((b[1]+b[0])/(E[1]+E[0]))*entropy(b[1], b[0])

def get_split(df, z):
    stop = df['Class'].value_counts()
    if len(stop) == 1:
        if df['Class'].iloc[0] == 0:
            return Node(0)
        else:
            return Node(1)

    if len(df.columns) < 2:
        if len(stop) == 2:
            if stop[0] > stop[1]:
                return Node(0)
            elif stop[1] > stop[0]:
                return Node(1)
            else:
                return Node(random.randint(0, 1))
        else:
            return Node(random.randint(0, 1))

    if df.shape[0] == 0:
        return Node(random.randint(0, 1))

    col = df.columns.values.tolist()
    if len(df.columns) == 2:
        root = Node(col[0])
        return root
    else:
        if z == 1:
            maxi = col[np.argmax([gain(df, x) for x in col[:-1]])]
            root = Node(maxi)
            return root
        else:
            maxi = col[np.argmax([gain2(df, x) for x in col[:-1]])]
            root = Node(maxi)
            return root

def build_tree(root, df0, z):
    if root.key == 1 or root.key == 0:
        return
    df1 = df0.loc[df0[root.key] == 0, df0.columns != root.key]
    df2 = df0.loc[df0[root.key] == 1, df0.columns != root.key]
    root.left = get_split(df1, z)
    root.right = get_split(df2, z)
    build_tree(root.left, df1, z)
    build_tree(root.right, df2, z)


def check(root, row):
    if root.isLeaf():
        if root.key == row['Class']:
            check.sum += 1
        return
    value = row[root.key]
    if value == 0:
        check(root.left, row)
    if value == 1:
        check(root.right, row)

def accuracy(root, df):
    check.sum = 0
    [check(root, df.iloc[x]) for x in range(df.shape[0])]
    return format((check.sum*100 / df.shape[0]), '.2f')


def print_tree(root):
    global n
    if root.isLeaf():
        print(root.key)
    for i in range(n):
        print("| ",end="")
    if root.left.isLeaf():
        print(str(root.key) + " = 0 : " + str(root.left.key))
    else:
        print(str(root.key) + " = 0 :")
        n += 1
        print_tree(root.left)
    if root.right.isLeaf():
        for i in range(n):
            print("| ",end="")
        print(str(root.key) + " = 1 : " + str(root.right.key))
        n -= 1
        return
    else:
        for i in range(n):
            print("| ",end="")
        print(str(root.key) + " = 1 :")
        n += 1
        print_tree(root.right)
        n -= 1



def copy_tree(root):
    copynode = None
    if (root):
        copynode = Node(root)
        copynode.key = root.key
        copynode.left = copy_tree(root.left)
        copynode.right = copy_tree(root.right)
    return copynode

def order_tree(root):
    if root is None:
        return

    if root.isLeaf():
        return

    if root.left is not None:
        order_tree(root.left)

    order_tree.order += 1
    root.order = order_tree.order

    if root.right is not None:
        order_tree(root.right)

def delete_tree(root, n):
    if root is None:
        return
    delete_tree(root.left, n)
    delete_tree(root.right, n)
    if root.order is None:
        if root.key == 0:
            delete_tree.zero += 1
        else:
            delete_tree.one += 1
    if root.order == n:
        if delete_tree.zero < delete_tree.one:
            root.key = 1
            root.order = None
        else:
            root.key = 0
            root.order = None
    root.left = None
    root.right = None

def findNode(root, n):
    if root is None:
        return
    elif n == root.order:
        return root
    elif n < root.order:
        return findNode(root.left, n)
    else:
        return findNode(root.right, n)

def pruning(root, df, l, k):
    dbest = root
    for i in range(l):
        d1 = copy_tree(root)
        m = random.randint(1, k)
        for j in range(m):
            order_tree.order = 0
            order_tree(d1)
            q = order_tree.order
            if q == 0:
                break
            p = random.randint(1, q)
            delete_tree.zero = 0
            delete_tree.one = 0
            node = findNode(d1, p)
            delete_tree(node, p)
        if accuracy(d1, df) > accuracy(dbest, df):
            dbest = d1
    return dbest


inputarg = sys.argv
if len(inputarg) < 7:
    print("more input arguments required")
elif not (inputarg[1].isdigit() and inputarg[2].isdigit()):
    print("arguments 1 and 2 should be integers")
else:
    l = int(inputarg[1])
    k = int(inputarg[2])
    data = pd.read_csv(inputarg[3])
    valid = pd.read_csv(inputarg[4])
    test = pd.read_csv(inputarg[5])
    n = 0

    if inputarg[6] == "no":
        top = get_split(data, 1)
        build_tree(top, data, 1)
        top2 = get_split(data, 2)
        build_tree(top2, data, 2)
        prun = pruning(top, valid, l, k)
        prun2 = pruning(top2, valid, l, k)
        print("accuracy of test dataset ID3 = " + str(accuracy(top, test)) + "%")
        print("accuracy of test dataset VImp = " + str(accuracy(top2, test)) + "%")
        print("accuracy of test dataset after pruning ID3 = " + str(accuracy(prun, test)) + "%")
        print("accuracy of test dataset after pruning VImp = " + str(accuracy(prun2, test)) + "%")
    elif inputarg[6] == "yes":
        top = get_split(data, 1)
        build_tree(top, data, 1)
        top2 = get_split(data, 2)
        build_tree(top2, data, 2)
        prun = pruning(top, valid, l, k)
        prun2 = pruning(top2, valid, l, k)
        print("accuracy of test dataset ID3 = " + str(accuracy(top, test)) + "%")
        print_tree(top)
        n = 0
        print("\naccuracy of test dataset VImp = " + str(accuracy(top2, test)) + "%")
        print_tree(top2)
        n = 0
        print("\naccuracy of test dataset after pruning ID3 = " + str(accuracy(prun, test)) + "%")
        print_tree(prun)
        n = 0
        print("\naccuracy of test dataset after pruning VImp = " + str(accuracy(prun2, test)) + "%")
        print_tree(prun2)
    else:
        print("argument 6 should be either 'yes' or 'no'")