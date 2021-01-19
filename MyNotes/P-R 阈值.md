### P-R 阈值 

|    &nbsp;     | 预测值(Positive) | 预测值(Negative) |
| :-----------: | :--------------: | :--------------: |
| 真实值(True)  |        TP        |        TN        |
| 真实值(False) |        FP        |        FN        |



 查准率`（P）`： $\frac{TP}{TP+FP}$ 

查全率`（R）`： $\frac{TP}{TP+TN}$ 

最优阈值的确定： 

+ 使用平衡点`（P = R）` 
+ $F_1$度量 :   $F_1=\frac{2PR}{P+R}$ (即$\frac{1}{F_1}=\frac{1}{2}\cdot(\frac{1}{P}+\frac{1}{R})$)，$F_1$是基于查准率与查全率的调和平均 
+ Fbeta：  $F_\beta=\frac{(1+\beta^2)\times P\times R}{(\beta^2\times P)+ R}$(即$\frac{1}{F_\beta}=\frac{1}{1 + \beta^2}\cdot(\frac{1}{P}+\frac{\beta^2}{R})$)，$F_\beta$是加权调和平均，其中$\beta > 0$度量了查全率对查准率的相对重要性，$\beta = 1$ 时退化为标准的$F_1$；    $\beta>1$时查全率有更大影响； $\beta < 1$时查准率有更大的影响。与算术平均（$\frac{(P+R)}{2}$）和几何平均（$\sqrt{P\times R}$）相比，调和平均更重视较小值