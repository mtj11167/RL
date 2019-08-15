# 强化学习遇到问题 #
> 根据D.Silver课程以及CS 294课程提出的问题
## D.Silver第三课 DP  ##
Q:DP的动态规划体现在哪里?  
A：根据 $$ v\_{\ast}(s)= \mathop{max}\limits\_{a\in A}(R\_{s}^{a}+\gamma\sum\_{s^{'}\in S}P\_{ss^{'}}^a v\_{\ast}(s^{'})) $$ 可以看出是动态规划， $$ v\_{\ast}(s) $$ 与 $$ v\_{\ast}(s^{'}) $$ 分别代表动态规划中的最优解

## D.Silver第四五课  model-free prediction 与 model-free control##
Q：为什么说将公式$$ \mu\_k\leftarrow \mu\_{k-1}+\frac 1k (x\_k - \mu\_{k-1})  $$改为$$\mu\_k\leftarrow \mu\_{k-1}+\alpha(x\_k - \mu\_{k-1})$$后，是不看重以前的回报了

A:可以证明，如果使用第二种方法，展开后式子变为$$ \mu\_k\leftarrow \alpha x\_k +\alpha(1-\alpha)x\_{k-1} +\alpha(1-\alpha)^2 x\_{k-2}+\cdots$$,如果使用第一种方法，展开后式子变为$$\mu\_k\leftarrow \frac 1k x\_k +\frac 1k x\_{k-1}+\frac 1k x\_{k-2}+\frac{k-3}{k} \bar x\_{k-3}$$
因此可以看出使用alpha的方式确实是对以前的x不看重了



Q：在AB例子中，为什么说TD用到了马尔科夫性
A：TD使用公式$$V(S\_t)\leftarrow V(S\_t)+\alpha(R\_{t+1}+\lambda V(S\_{t+1})-V(S\_{t+1})))$$ ，其中的$$R\_{t+1}+\lambda V(S\_{t+1})$$ 项其实是贝尔曼方程，而贝尔曼方程的推导中使用到了马尔可夫性，因此说使用了马尔可夫性。

