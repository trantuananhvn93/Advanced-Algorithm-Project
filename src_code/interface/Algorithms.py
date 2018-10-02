import numpy as np
import datetime as d
import math
import queue
import time
from random import randint
import matplotlib.pyplot as plt


s1 = list();
p1 = list();

def edit_distance(A, B):
    n = A.size
    m = B.size
    ED = np.zeros((n + 1, m + 1), dtype='int32')
    ptr = np.zeros((n + 1, m + 1), dtype='int32')

    for i in range(n + 1):  # attention:the range is from 0 to n
        ED[i, 0] = i
        if i > 0:
            ptr[i, 0] = 4  # up
    for j in range(m + 1):
        ED[0, j] = j
        if j > 0:
            ptr[0, j] = 2  # left

    if (n == 0 or m == 0):
        A2, B2 = alignment(A, B, ptr)
        return ED, ptr, ED[n, m], A2, B2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # MATRIX ED
            diff = 0 if A[i - 1] == B[j - 1] else 1
            ED[i, j] = min(ED[i - 1, j] + 1, ED[i, j - 1] + 1, ED[i - 1, j - 1] + diff)

            # TRACE-BACK
            if (ED[i, j] == ED[i - 1, j] + 1):  # UP : DELETION
                ptr[i, j] = ptr[i, j] | 4
            if (ED[i, j] == ED[i, j - 1] + 1):  # lEFT : INSERTION
                ptr[i, j] = ptr[i, j] | 2
            if (ED[i, j] == ED[i - 1, j - 1] + diff):  # DIAGONAL : SUBSTITUTION
                ptr[i, j] = ptr[i, j] | 1
    edInt = ED[n, m]
    A2, B2 = alignment(A, B, ptr)
    return ED, ptr, edInt, A2, B2


#edit d wo trb

# Dynamic Programming Method
def edit_distance_wotr(str_A, str_B):
    A = np.fromstring(str_A, dtype='|S1')
    B = np.fromstring(str_B, dtype='|S1')
    n = A.size
    m = B.size
    ED = np.zeros((n + 1, m + 1), dtype='int32')

    for i in range(m+1):
        ED[0,i] = i

    for j in range(n+1):
        ED[j,0] = j

    for k in range (1,n+1):
        for l in range(1,m+1):
            if A[k-1] == B[l-1]:
                ED[k,l] = ED[k-1, l-1]

            else:
                ED[k,l] = min(ED[k-1, l-1], ED[k, l-1], ED[k-1, l]) + 1

    #traceback
    tr = [] #list of actions
    tr1 = []
    p=n
    q=m
    while p>=0 or q>=0:
        if p>0 and q>0:
            if A[p-1] == B[q-1]:
                tr1.append('D')
                p= p-1
                q = q-1
            else:
                if ED[p-1,q-1] == min(ED[p-1,q-1],ED[p,q-1],ED[p-1,q]):
                    tr.append("Substitute {0} with {1}".format((A[p-1]).decode("utf-8"), (B[q-1]).decode("utf-8")))
                    tr1.append('D')  # SUBSTITUTION
                    p = p - 1
                    q = q - 1
                elif ED[p,q-1] == min(ED[p-1,q-1],ED[p,q-1],ED[p-1,q]):
                    tr.append("Insert {0}".format((B[q-1]).decode("utf-8")))
                    tr1.append('L')  # INSERTION
                    q=q-1
                else:
                    tr.append("Remove {0}".format((A[p-1]).decode("utf-8")))
                    tr1.append('U')  # DELETION
                    p=p-1
        elif p==0 and q>0:
            tr.append("Insert {0}".format((B[q - 1]).decode("utf-8")))
            tr1.append('L')
            q=q-1
        elif p>0 and q==0:
            tr.append("Remove {0}".format((A[p - 1]).decode("utf-8")))
            tr1.append('U')
            p=p-1
        else:
            p=-1
            q=-1
    tr.reverse()

    #   Update A and B with alignment
    A2 = []
    B2 = []
    k = 0
    h = 0
    for i in range(len(tr1), 0, -1):
        if tr1[i - 1] == 'D':
            A2.append(A[k])
            B2.append(B[h])
            k += 1
            h += 1
        elif tr1[i - 1] == 'U':
            A2.append(A[k])
            B2.append('-')
            k += 1
        elif tr1[i - 1] == 'L':
            A2.append('-')
            B2.append(B[h])
            h += 1

    ed = ED[n,m]  #edit distance

    return ED,ed,tr, A2, B2


def LevenshteinDistance(string1, string2):
    n1=len(string1)
    n2=len(string2)
    if string1==string2:
        return 0
    if n1==0 or n2==0:
        return n1+n2
    #The d matriz will hold the Leventshtein distance
    d = []
    for i in range(n1+1):
        row = []
        for j in range(n2+1):
            if i==0 or j==0:
                row.append(i+j)                 #target prefixes
            else:
                if string1[i-1]==string2[j-1]:  #same letter, no operation
                    row.append(d[i-1][j-1])
                else:
                    minimo=min(d[i-1][j] + 1,   #a deletion
                               row[j-1]+1,      #an insertion
                               d[i-1][j-1] + 1) #a substitution
                    row.append(minimo)
        d.append(row)
    return d[n1][n2], d, n1, n2


def ed_greedy(a, b):
    value = 0
    if len(a) > len(b):
        bigger = a;
        smaller = b
    else:
        bigger = b;
        smaller = a
    BIGGER = []
    SMALLER = []
    for i in range(len(bigger) - len(smaller)):
        SMALLER.append(0)

    i = 0
    while i < len(bigger):
        BIGGER.append(bigger[i]);
        i += 1
    i = 0
    while i < len(smaller):
        SMALLER.append(smaller[i]);
        i += 1

    for i in range(len(BIGGER)):
        if (BIGGER[-1 - i]) != (SMALLER[-1 - i]):
            value += 1

    return value



#greedy method accepting only charactor values
def editDistanceGreedy(string1, string2):
    rnd = 0
    while len(longestSubstringFinder(string1,string2))>1:
        lcs = longestSubstringFinder(string1, string2)
        string1 = string1.replace(lcs, str(rnd))
        string2 = string2.replace(lcs, str(rnd))
        rnd += 1

    return edit_distance_DynamicForGreedy(string1, string2)






def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

def edit_distance_DynamicForGreedy(str_A, str_B):
    A = np.fromstring(str_A, dtype='|S1')
    B = np.fromstring(str_B, dtype='|S1')
    n = A.size
    m = B.size
    ED = np.zeros((n + 1, m + 1), dtype='int32')

    for i in range(m + 1):
        ED[0, i] = i

    for j in range(n + 1):
        ED[j, 0] = j

    for k in range(1, n + 1):
        for l in range(1, m + 1):
            if A[k - 1] == B[l - 1]:
                ED[k, l] = ED[k - 1, l - 1]

            else:
                ED[k, l] = min(ED[k - 1, l - 1], ED[k, l - 1], ED[k - 1, l]) + 1

    return ED[n, m]  # edit distance


def linear_space_Hirshberg(A, B):
    n = len(A)
    m = len(B)

    # position to divide String A
    h = round(n / 2)
    # init ED 1D array
    ED = np.zeros((m + 1), dtype='int32')
    # init Hirshberg 1D array
    H = np.zeros((m + 1), dtype='int32')

    for j in range(m + 1):
        ED[j] = j

    for i in range(1, n + 1):
        # set up the left and Diagonal value of ED on each new row
        left = i
        dag = i - 1
        # Left and Diagonal value of H on each new row
        H_left = 0
        H_dag = 0
        for j in range(1, m + 1):
            # Compute ED
            diff = 0 if A[i - 1] == B[j - 1] else 1
            curr = min(ED[j] + 1, left + 1, dag + diff)
            # Compute Array 1D H thanks to result of ED
            if i == h:
                H[j] = j
            if (i > h):
                if (curr == left + 1):
                    H_curr = H_left
                elif (curr == ED[j] + 1):
                    H_curr = H[j]
                else:
                    H_curr = H_dag
                # Update new value of left, dag and curr
                H_left = H_curr
                H_dag = H[j]
                H[j] = H_curr
            # Update ED array
            left = curr
            dag = ED[j]
            ED[j] = curr

    return np.array([h, H[m]], dtype='int32')


def Divide_and_Conquer_ED(A, B, newED=None, newA=None, newB=None):
    n = len(A)
    m = len(B)
    s2 = list()
    # global lists:
    if newED is None:
        newED = []
    if newA is None:
        newA = []
    if newB is None:
        newB = []

    if (n < 2 or m < 2):

        ED, _A, _B = edit_distanceForDnD(A, B)

        newED.append(ED)
        for word in _A:
            newA.append(word)
        for word in _B:
            newB.append(word)

    else:
        H = linear_space_Hirshberg(A, B)
        s2.append(H[0])
        s2.append(H[1])
        s2.append(A[:H[0]])
        s2.append(B[:H[1]])
        s2.append(A[H[0]:])
        s2.append(B[H[1]:])
        s1.append(s2)
        Divide_and_Conquer_ED(A[:H[0]], B[:H[1]], newED, newA, newB)
        Divide_and_Conquer_ED(A[H[0]:], B[H[1]:], newED, newA, newB)

    return np.sum(np.array(newED)), newA, newB, s1


def stripe_edit_distance(A, B):
    n = len(A)
    m = len(B)

    ED = np.empty((n + 1, m + 1))
    ptr = np.zeros((n + 1, m + 1), dtype='int32')

    if (n == 0 or m == 0):
        for i in range(n + 1):
            ED[i, 0] = i
            if i > 0:
                ptr[i, 0] = 4  # up
        for j in range(m + 1):
            ED[0, j] = j
            if j > 0:
                ptr[0, j] = 2  # left

    else:

        ED[:] = math.inf  # infinity

        # auto set up the threshold K
        k = 1
        while (abs(n - m) > k):
            k += 1
        # print("k = ",k )

        for j in range(m + 1):
            if j < k + 1:
                ED[0, j] = j
                ptr[0, j] = 2 if j > 0 else 0  # left

        for i in range(n + 1):
            if i < k + 1:
                ED[i, 0] = i
                ptr[i, 0] = 4 if i > 0 else 0  # Up

        for i in range(1, n + 1):
            # set up the threshold window
            a = max(1, i - k)
            b = min(m + 1, i + k + 1)

            for j in range(a, b):
                # MATRIX ED
                diff = 0 if A[i - 1] == B[j - 1] else 1
                ED[i, j] = min(ED[i - 1, j] + 1, ED[i, j - 1] + 1, ED[i - 1, j - 1] + diff)

                # TRACE-BACK
                if (ED[i, j] == ED[i - 1, j] + 1):  # UP : DELETION
                    ptr[i, j] = ptr[i, j] | 4
                if (ED[i, j] == ED[i, j - 1] + 1):  # lEFT : INSERTION
                    ptr[i, j] = ptr[i, j] | 2
                if (ED[i, j] == ED[i - 1, j - 1] + diff):  # DIAGONAL : SUBSTITUTION
                    ptr[i, j] = ptr[i, j] | 1

    A2, B2 = alignment(A, B, ptr)
    return ED[n, m], A2, B2, ED, ptr, k


def alignment(A, B, ptr, L=None):
    # create a trackback line
    if L is None:
        L = []
        row = ptr.shape[0] - 1
        col = ptr.shape[1] - 1
        while (row != 0 or col != 0):
            if (ptr[row, col] == 1) or (ptr[row, col] == 7):
                L.append('D')
                row = row - 1
                col = col - 1
            elif (ptr[row, col] == 4) or (ptr[row, col] == 5) or (ptr[row, col] == 6):
                L.append('U')
                row = row - 1
            elif (ptr[row, col] == 2) or (ptr[row, col] == 3):
                L.append('L')
                col = col - 1
            else:
                print("------ERROR----")
                print("ptr:\n", ptr, "\nwhere ptr=", ptr[row, col], "row=", row, "col=", col)
                return [], []


                # Update A and B with alignment
    A2 = []
    B2 = []
    k = 0
    h = 0
    for i in range(len(L), 0, -1):
        if L[i - 1] == 'D':
            A2.append(A[k])
            B2.append(B[h])
            k += 1
            h += 1
        elif L[i - 1] == 'U':
            A2.append(A[k])
            B2.append('-')
            k += 1
        elif L[i - 1] == 'L':
            A2.append('-')
            B2.append(B[h])
            h += 1

    return A2, B2


def ed_recursive(a, b, path=[]):
    # The function return a tuple (value of distance, path to follow (left,mid,right, at each step) to reach a solution)

    if a == "":
        return (len(b), path + ['m'] * len(b))

    if b == "":
        return (len(a), path + ['g'] * len(a))

    if a[-1] == b[-1]:
        cost = 0
    else:
        cost = 1

    gauche = ed_recursive(a[:-1], b, path + ['g'])
    gauche = (gauche[0] + 1, gauche[1])

    milieu = ed_recursive(a, b[:-1], path + ['m'])
    milieu = (milieu[0] + 1, milieu[1])

    droite = ed_recursive(a[:-1], b[:-1], path + ['d'])
    droite = (droite[0] + cost, droite[1])

    res = min([gauche, milieu, droite], key=lambda x: x[0])

    return res


def h(a, b):
    return abs(len(a) - len(
        b))  # This corresponds to h in f=g*+h seen in the courses, that return the "at least needed operations to reach goal" for a given node


def cutoff(a, b, bound,
           cumulcost):  # This is the function that indicates if we have or not to prune a node and his subtree
    if cumulcost + h(a, b) >= bound:
        return True
    else:
        return False


def ed_bb(a, b, cumulcost=0, path=[], bound=None):
    # The function return a tuple (value of distance, path to follow (left,mid,right, at each step) to reach a solution)
    # We browse the tree in depth first search
    # In each step we can choose 3 branches :
    # The two ones that respectively indicates to look at (a[:-1],b) and (a,b[:-1]) and cost one
    # The last one that indicate to look for (a[:-1],b[:-1]), cost zero or one


    if bound is None:
        bound = []
        bound.append(
            987654321)  # As the bound can't be used as a local int variable in this algorithm, I used this trick (to avoid to define a global variable)

    if a == "":
        if cumulcost + len(b) < bound[0]:
            bound[0] = cumulcost + len(
                b)  # When reaching a solution, we check if the cost is lesser than current bound, and if so we update the bound
        return (len(b), path + ['m'] * len(b))

    if b == "":
        if cumulcost + len(a) < bound[0]:
            bound[0] = cumulcost + len(a)
        return (len(a), path + ['g'] * len(a))

    if a[-1] == b[-1]:
        cost = 0
    else:
        cost = 1

    if cutoff(a, b, bound[0], cumulcost) == False:
        gauche = ed_bb(a[:-1], b, cumulcost + 1, path + ['g'],
                       bound)  # The steps here are classical as in the recursive way
        gauche = (gauche[0] + 1, gauche[1])

        milieu = ed_bb(a, b[:-1], cumulcost + 1, path + ['m'], bound)
        milieu = (milieu[0] + 1, milieu[1])

        droite = ed_bb(a[:-1], b[:-1], cumulcost + cost, path + ['d'], bound)
        droite = (droite[0] + cost, droite[1])

        res = min([gauche, milieu, droite], key=lambda x: x[0])

        return res

    else:
        return (123456789, path)  # This permits to return an (int,path) in every case, otherwise, the function cannot works. Moreother if we reach this step it means that cutoff occured, so first we don't make recursive call, and second we return a very big number that cannot be the min of the compared paths


def ed_bb_with_path_and_alignment(a,b):
    return (ed_bb(a,b), alignment_bb(a,b))


#special
def edit_distanceForDnD(A, B):
    n = len(A)
    m = len(B)
    ED = np.zeros((n + 1, m + 1), dtype='int32')
    ptr = np.zeros((n + 1, m + 1), dtype='int32')

    for i in range(n + 1):  # attention:the range is from 0 to n
        ED[i, 0] = i
        if i > 0:
            ptr[i, 0] = 4  # up
    for j in range(m + 1):
        ED[0, j] = j
        if j > 0:
            ptr[0, j] = 2  # left

    if (n == 0 or m == 0):
        A2, B2 = alignment(A, B, ptr)
        return ED[n, m], A2, B2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # MATRIX ED
            diff = 0 if A[i - 1] == B[j - 1] else 1
            ED[i, j] = min(ED[i - 1, j] + 1, ED[i, j - 1] + 1, ED[i - 1, j - 1] + diff)

            # TRACE-BACK
            if (ED[i, j] == ED[i - 1, j] + 1):  # UP : DELETION
                ptr[i, j] = ptr[i, j] | 4
            if (ED[i, j] == ED[i, j - 1] + 1):  # lEFT : INSERTION
                ptr[i, j] = ptr[i, j] | 2
            if (ED[i, j] == ED[i - 1, j - 1] + diff):  # DIAGONAL : SUBSTITUTION
                ptr[i, j] = ptr[i, j] | 1

    A2, B2 = alignment(A, B, ptr)
    return ED[n, m], A2, B2


def rec_alignment(a, b):
    A = []
    B = []
    AA = []  # Simply will containt characters of string a
    BB = []  # Simply will containt characters of string b
    i = 0
    while i < len(a):
        AA.append(a[i]);
        i += 1
    i = 0
    while i < len(b):
        BB.append(b[i]);
        i += 1

    path = ed_recursive(a, b)[1]
    path.reverse()  # We reverse path because it is more conveniant for indexation

    for i in range(len(path)):

        if path[i] == 'g':
            B.append('*')
        else:
            B.append(BB[0]);
            del BB[0]
        if path[i] == 'm':
            A.append('*')
        else:
            A.append(AA[0]);
            del AA[0]
    return (A, B)


def rec_ed_with_path_and_alignment(a, b):
    return (ed_recursive(a, b), rec_alignment(a, b))


def alignment_bb(a, b):
    A = []
    B = []
    AA = []  # Simply will containt characters of string a
    BB = []  # Simply will containt characters of string b
    i = 0
    while i < len(a):
        AA.append(a[i]);
        i += 1
    i = 0
    while i < len(b):
        BB.append(b[i]);
        i += 1

    path = ed_bb(a, b)[1]
    path.reverse()  # We reverse path because it is more conveniant for indexation

    for i in range(len(path)):

        if path[i] == 'g':
            B.append('*')
        else:
            B.append(BB[0]);
            del BB[0]
        if path[i] == 'm':
            A.append('*')
        else:
            A.append(AA[0]);
            del AA[0]
    return (A, B)



#ffff
def divide_conquer_version2(A, B):
    def edit_distance_forwardv2(A, B):
        n = len(A)
        m = len(B)
        ED = np.zeros((m + 1), dtype='int32')

        for j in range(m + 1):
            ED[j] = j

        for i in range(1, n + 1):
            left = i
            dag = i - 1
            ED[0] = i
            for j in range(1, m + 1):
                # MATRIX ED
                diff = 0 if A[i - 1] == B[j - 1] else 1
                curr = min(ED[j] + 1, left + 1, dag + diff)
                left = curr
                dag = ED[j]
                ED[j] = curr

        return ED

    def edit_distance_backwardv2(A, B):
        n = len(A)
        m = len(B)

        ED = np.zeros((m + 1), dtype='int32')

        for j in range(m, -1, -1):
            ED[j] = m - j

        for i in range(n - 1, -1, -1):
            left = n - i
            dag = n - i - 1
            ED[m] = n - i
            for j in range(m - 1, -1, -1):
                # MATRIX ED
                diff = 0 if A[i] == B[j] else 1
                curr = min(ED[j] + 1, left + 1, dag + diff)
                left = curr
                dag = ED[j]
                ED[j] = curr

        return ED

    def find_min_pointv2(row):

        # find the minimal value of the row
        min_value = min(row)
        # find index of all minimums
        min_index = np.where(row == min_value)[0]

        return min_index[0]

    def Hirshbergv2(A, B):
        n = len(A)
        m = len(B)

        # position to divide String A
        h = int(n / 2)

        ED_forward = edit_distance_forwardv2(A[:h], B)
        ED_backward = edit_distance_backwardv2(A[h:], B)

        # Adding corresponding elements of these two rows h
        row_h = ED_forward[:] + ED_backward[:]

        # position to divide String B
        h2 = find_min_pointv2(row_h)

        return np.array([h, h2], dtype='int32')

    def Divide_and_Conquerv2(A, B, newED=None, newA=None, newB=None):
        n = len(A)
        m = len(B)
        p2 = list()

        # global lists:
        if newED is None:
            newED = []
        if newA is None:
            newA = []
        if newB is None:
            newB = []

        if (n < 2 or m < 2):

            ED, _A, _B = edit_distanceForDnD(A, B)

            newED.append(ED)
            for word in _A:
                newA.append(word)
            for word in _B:
                newB.append(word)

        else:
            H = Hirshbergv2(A, B)
            p2.append(H[0])
            p2.append(H[1])
            p2.append(A[:H[0]])
            p2.append(B[:H[1]])
            p2.append(A[H[0]:])
            p2.append(B[H[1]:])
            p1.append(p2)
            Divide_and_Conquerv2(A[:H[0]], B[:H[1]], newED, newA, newB)
            Divide_and_Conquerv2(A[H[0]:], B[H[1]:], newED, newA, newB)

        return np.sum(np.array(newED)), ''.join(newA), ''.join(newB)

    ed, _A, _B = Divide_and_Conquerv2(A, B)
    return ed, _A, _B, p1