import pandas as pd
# import numpy as np
# import sys
from itertools import combinations


# from collections import Counter
from IPython.display import display


def aproiri_gen(orders, min_support=0.15, max_len=4):
    # supp will store the support of an itemset
    supp = {}
    L = set(orders.columns)

    # Generating combinations of items
    for i in range(1, max_len + 1):
        A = list(combinations(L, i))
        L = set()  # reset L for next (i+1) iteration

        for j in list(A):
            support = orders.loc[:, j].product(axis=1).sum() / len(orders.index)
            if support >= min_support:
                supp[j] = support

                L = set(set(L) | set(j))
    B = pd.DataFrame(list(supp.items()), columns=["Items", "Support"])
    return (B)


def association_rule(df, min_threshold=0.5):
    import pandas as pd
    from itertools import permutations

    # STEP 1:
    # creating required varaible
    support = pd.Series(df.Support.values, index=df.Items).to_dict()
    data = []
    L = df.Items.values

    # Step 2:
    # generating rule using permutation
    p = list(permutations(L, 2))

    # Iterating through each rule
    for i in p:

        # If LHS(Antecedent) of rule is subset of RHS then valid rule.
        if set(i[0]).issubset(i[1]):
            conf = support[i[1]] / support[i[0]]
            # print(i, conf)
            if conf > min_threshold:
                # print(i, conf)
                j = i[1][not i[1].index(i[0][0])]
                lift = support[i[1]] / (support[i[0]] * support[(j,)])
                leverage = support[i[1]] - (support[i[0]] * support[(j,)])
                try:
                    convection = (1 - support[(j,)]) / (1 - conf)
                except ZeroDivisionError:
                    convection = 'inf'
                data.append([i[0], (j,), support[i[0]], support[(j,)], support[i[1]], conf, lift, leverage, convection])

    # STEP 3:
    result = pd.DataFrame(data, columns=["antecedents", "consequents", "antecedent support", "consequent support",
                                         "support", "confidence", "Lift", "Leverage", "Convection"])
    return (result)


orders = pd.read_csv('sampledata1.csv', names=['Products'], header=None, sep=';')

# generation of tidy dataset
data = list(orders["Products"].apply(lambda x: x.split(',')))

from mlxtend.preprocessing import TransactionEncoder

ten = TransactionEncoder()
ten_data = ten.fit(data).transform(data)
orders = pd.DataFrame(ten_data, columns=ten.columns_).astype(int)

freq_itemset = aproiri_gen(orders)
freq_itemset = freq_itemset.sort_values(by='Support', ascending=False)
print(freq_itemset)

my_rule = association_rule(freq_itemset, 0.5)

#my_rule = my_rule.sort_values(by='Lift', ascending=False).head(10)
print('###################################################')
display(my_rule.sort_values(by='Lift', ascending=False).head(10))
