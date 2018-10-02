from flask import Flask as fl, render_template
import numpy as np
import datetime as d
import math
import Algorithms
import Algorithms2

app = fl(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/dynamic/')
@app.route('/dynamic/<str_A>/<str_B>')
def dynamic(str_A=None, str_B=None):
    if str_A:
        A = np.fromstring(str_A, dtype='|S1')
        B = np.fromstring(str_B, dtype='|S1')
        ED1, ptrl, edInt, newA, newB = Algorithms.edit_distance(A, B)
        ED1 = ED1.tolist()
        ptrl = ptrl.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)

        for idx,zz in enumerate(newA):
            if newA[idx] == '-':
                newA[idx] = newA[idx]
            else:
                newA[idx] = newA[idx].decode('utf-8')

        for idx,zz in enumerate(newB):
            if newB[idx] == '-':
                newB[idx] = newB[idx]
            else:
                newB[idx] = newB[idx].decode('utf-8')

        return render_template("layout.html", EDI=ED1, PTR=ptrl, str_A=str_A, str_B=str_B, arr_A=arr_A, arr_B=arr_B, edInt=edInt, newA=newA, newB=newB)
    else:
        A = np.fromstring("", dtype='|S1')
        B = np.fromstring("", dtype='|S1')
        ED1, ptrl, edInt, newA, newB = Algorithms.edit_distance(A, B)
        ED1 = ED1.tolist()
        ptrl = ptrl.tolist()
        return render_template("layout.html", EDI=ED1, PTR=ptrl, str_A=str_A, str_B=str_B)

@app.route('/dynamicwithouttraceback/')
@app.route('/dynamicwithouttraceback/<str_A>/<str_B>')
def dynamic_wotr(str_A=None, str_B=None):
    if str_A:
        ED2, ed2, tr2, newA, newB = Algorithms.edit_distance_wotr(str_A, str_B)
        ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)

        for idx,zz in enumerate(newA):
            if newA[idx] == '-':
                newA[idx] = newA[idx]
            else:
                newA[idx] = newA[idx].decode('utf-8')

        for idx,zz in enumerate(newB):
            if newB[idx] == '-':
                newB[idx] = newB[idx]
            else:
                newB[idx] = newB[idx].decode('utf-8')

        return render_template("dynamicwithouttraceback.html", EDI=ED2, tr2=tr2, str_A=str_A, str_B=str_B, arr_A=arr_A, arr_B=arr_B, edInt=ed2, newA=newA, newB=newB)
    else:
        return render_template("dynamicwithouttraceback.html", str_A=str_A, str_B=str_B)



@app.route('/dynamicalternative/')
@app.route('/dynamicalternative/<str_A>/<str_B>')
def dynamic_alt(str_A=None, str_B=None):
    if str_A:
        ed2 , ED2, n1, n2 = Algorithms.LevenshteinDistance(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("dynamicalternative.html", EDI=ED2, str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B)
    else:
        return render_template("dynamicalternative.html", str_A=str_A, str_B=str_B)



@app.route('/greedy/')
@app.route('/greedy/<str_A>/<str_B>')
def greedy(str_A=None, str_B=None):
    if str_A:
        ed2 = Algorithms.ed_greedy(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("greedy.html", str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B)
    else:
        return render_template("greedy.html", str_A=str_A, str_B=str_B)

@app.route('/greedyalternative/')
@app.route('/greedyalternative/<str_A>/<str_B>')
def greedyalt(str_A=None, str_B=None):
    if str_A:
        ed2 = Algorithms.editDistanceGreedy(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("greedyalt.html", str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B)
    else:
        return render_template("greedyalt.html", str_A=str_A, str_B=str_B)

@app.route('/divideandconquer/')
@app.route('/divideandconquer/<str_A>/<str_B>')
def dividenc(str_A=None, str_B=None):
    if str_A:
        ed2, newA, newB, s1 = Algorithms.Divide_and_Conquer_ED(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("divideandconquer.html", str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B, newA=newA, newB=newB, s1=s1)
    else:
        return render_template("divideandconquer.html", str_A=str_A, str_B=str_B)



@app.route('/divideandconquer2/')
@app.route('/divideandconquer2/<str_A>/<str_B>')
def dividencrv2(str_A=None, str_B=None):
    if str_A:
        ed2, newA, newB, s1 = Algorithms.divide_conquer_version2(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("divideandconquer2.html", str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B, newA=newA, newB=newB, s1=s1)
    else:
        return render_template("divideandconquer2.html", str_A=str_A, str_B=str_B)




@app.route('/stripek/')
@app.route('/stripek/<str_A>/<str_B>')
def stripe(str_A=None, str_B=None):
    if str_A:
        ed2, newA, newB, ED1, ptrl, k = Algorithms.stripe_edit_distance(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("stripek.html", str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B, newA=newA, newB=newB, EDI=ED1, PTR=ptrl, k=k)
    else:
        return render_template("stripek.html", str_A=str_A, str_B=str_B)


@app.route('/recursive/')
@app.route('/recursive/<str_A>/<str_B>')
def recursive(str_A=None, str_B=None):
    if str_A:
        ed2, (newA, newB) = Algorithms.rec_ed_with_path_and_alignment(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("recursive.html", str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B, newA=newA, newB=newB)
    else:
        return render_template("recursive.html", str_A=str_A, str_B=str_B)


@app.route('/branchandboundwithpath/')
@app.route('/branchandboundwithpath/<str_A>/<str_B>')
def branchandboundwithpath(str_A=None, str_B=None):
    if str_A:
        ed2, (newA, newB) = Algorithms.ed_bb_with_path_and_alignment(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("branchandboundwithpath.html", str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B, newA=newA, newB=newB)
    else:
        return render_template("branchandboundwithpath.html", str_A=str_A, str_B=str_B)



@app.route('/branchandboundwithqueue/')
@app.route('/branchandboundwithqueue/<str_A>/<str_B>')
def branchandboundwithqueue(str_A=None, str_B=None):
    if str_A:
        ed2, newA, newB = Algorithms2.Branch_and_Bound(str_A, str_B)
        # ED2 = ED2.tolist()
        arr_A = list(str_A)
        arr_B = list(str_B)
        return render_template("branchandboundwithqueue.html", str_A=str_A, str_B=str_B, edInt=ed2, arr_A=arr_A, arr_B=arr_B, newA=newA, newB=newB)
    else:
        return render_template("branchandboundwithqueue.html", str_A=str_A, str_B=str_B)



@app.route('/proteindb/')
def proteindb():
        return render_template("proteindb.html")


if __name__ == "__main__":
    app.run()

