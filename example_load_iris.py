# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:18:26 2018

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.tree


if __name__ == '__main__':
    # irisという名前の有名なデータセットがある。それを読み込むための関数が提供されている。
    # http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
    iris_dataset = sklearn.datasets.load_iris()

    # 入力データとラベルデータを取り出す。
    data = iris_dataset.data  # 入力 （又は「説明変数」や「x」などと呼ばれることもある）。
    target = iris_dataset.target  # ラベル （又は「目的変数」や「t」などと呼ばれることもある）。

    # 「決定木」という手法による分類器を使う。
    model = sklearn.tree.DecisionTreeClassifier()

    # fitメソッドに入力データとラベルデータを与えると、学習が実行される。
    # なお、このメソッドに戻り値はない。つまり、習結果を戻り値として返すのではなく、
    # このメソッドを呼ぶと model が直接書き換わるので、注意が必要である。
    model.fit(data, target)

    # predictメソッドに入力データを与えると、予測値が出力される。
    pred = model.predict(data)

    # 分類結果である予測クラスと真のクラスを見比べてみる。
    # 今回のケースでは、全く同じになっているはずだ。これは
    print('Predicted class:', pred)
    print('True class:     ', target)
    print()

    # 正解率（正解事例の個数 / 全事例の個数）を計算するには、
    # score = np.average(pred == target)
    # とすればよいが、以下のような便利メソッドもある
    score = model.score(data, target)
    print('Accuracy rate:', score)

    # 決定木の構造をグラフとして出力するための関数。
    # http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz
    # 第二引数に出力ファイル名を指定でき、もし省略した場合には `tree.dot` という名前のファイルが出力される。
    # 拡張子の `.dot` は、グラフ構造を記述するための「DOT言語」のファイルであることを表している。
    # dot形式で記述されたデータを図として描画するための方法は色々あるが、今回は以下のサイトを使う。
    # WebGraphviz　http://www.webgraphviz.com/
    # 出力されたdotファイルをエディタで開き、その内容をWebGraphvizのテキストボックスに貼り付けて、
    # `Generate Graph!` ボタンを押すと図が生成される。
    # あるいは、ローカルPC上でdotファイルをpdfファイルに変換したければ、Graphvizをインストールしたうえで、
    # dot -T pdf tree.dot -o tree.pdf
    # というコマンドを実行すればよい。
    sklearn.tree.export_graphviz(model, 'tree.dot')

    # 各予測が、木のどの経路（path）を辿ったのかを見ることもできる。
    # Spyderの「変数エクスプローラー」で、変数`path`を表示してみよう。
    # pathの各行（各横ベクトル）が、各データ点の経路を表している。
    # 木のグラフ構造の各ノードに、根→葉、左→右の順に番号が振られており、
    # pathの列番号がノード番号に対応している。通ったノードに1が入っている。
    path = model.decision_path(data).todense()
