## DEEP Learning---几个求导公式




$$
\begin{align}
L(W) &= \frac{1}{2m}\sum_{i=1}^m\lVert(h(x_i)-y_i)\rVert^2\\
	 &= \frac{1}{2}\lVert f(Wx+b)-y\rVert^2
\end{align}
$$




1. 
   
   $$
   y = Wx\\
    \nabla_yf\stackrel{?}{\longrightarrow}\nabla_wf\\
    f(y)\\
    \left[
    \begin{matrix}
    w_{11}&w_{12}&\cdots&w_{1n}\\
    w_{21}&w_{22}&\cdots&w_{2n}\\
    \cdots&\cdots&\cdots&\cdots\\
    w_{m1}&w_{m2}&\cdots&w_{mn}\\
    \end{matrix}
    \right]
    \left[
    \begin{matrix}
    x_1\\
    x_2\\
    \cdots\\
    x_n\\
    \end{matrix}
    \right]
    =
    \left[
    \begin{matrix}
    y_1\\
    y_2\\
    \cdots\\
    y_m\\
    \end{matrix}
    \right]
    \\
 \frac{\partial{f}}{\partial{w_{ij}}} = \frac{\partial{f}}{\partial{y_i}}x_j
   $$
   **结论：**$\nabla_wf = (\nabla_yf)x^T$

2. 
   $$
   y = Wx\\
      \nabla_yf\stackrel{?}{\longrightarrow}\nabla_xf\\
      f(y)\\
      \frac{\partial{f}}{\partial{x_i}} = \sum\limits_{j=1}^m\frac{\partial{f}}{\partial{y_i}}w_{ji} = \nabla_yf [w_{1i}\;\cdots\;w_{mi}]\\
      即：\nabla_xf = W^T\nabla_yf
   $$
   



3. 

   **激活函数作用:**
   $$
   y = g(x)\\\
      y_i = g(x_i)\\
      f(y)\\
      \nabla_yf\stackrel{?}{\longrightarrow}\nabla_xf\\
      \frac{\partial{f}}{\partial{x_i}} = \frac{\partial{f}}{\partial{y_i}}\frac{\partial{y_i}}{\partial{x_i}}\\
   $$
      **结论：**$\nabla_xf = \nabla_yf\odot g^\prime(x)$

4. 
   $$
   加权：u = Wx\\
      激活：y = g(u)\\
      f(y)\\
      \nabla_yf\stackrel{?}{\longrightarrow}\nabla_xf\\
      结论：\nabla_xf = W^T(\nabla_uf) = W^T((\nabla_yf)\odot g^\prime(u))
   $$
   



5. 
   $$
    y = g(x)\\
      y_i = g_i(x_1,x_2,\cdots,x_n),i=1,\cdots,m\\
      \nabla_yf\stackrel{?}{\longrightarrow}\nabla_xf\\
      \\
      \frac{\partial{f}}{\partial{x_j}} = \sum_{i=1}^m{\frac{\partial{f}}{\partial{y_i}}\frac{\partial{y_i}}{\partial{x_j}}} = [\frac{\partial{y_1}}{\partial{x_j}}\;\cdots\;\frac{\partial{y_m}}{\partial{x_j}}]
      \left[
      \begin{matrix}
      \frac{\partial{f}}{\partial{y_1}}\\
      \cdots\\
      \frac{\partial{f}}{\partial{y_m}}\\
      \end{matrix}
      \right]
      \\
      \left[
      \begin{matrix}
      \frac{\partial{f}}{\partial{x_1}}\\
      \cdots\\
      \frac{\partial{f}}{\partial{x_n}}\\
      \end{matrix}
      \right]
      =
      \left[
      \begin{matrix}
      \frac{\partial{y_1}}{\partial{x_1}}&\cdots&\frac{\partial{y_m}}{\partial{x_1}}\\
      \cdots&\cdots&\cdots\\
      \frac{\partial{y_1}}{\partial{x_n}}&\cdots&\frac{\partial{y_m}}{\partial{x_n}}\\
      \end{matrix}
      \right]
      \left[
      \begin{matrix}
      \frac{\partial{f}}{\partial{y_1}}\\
      \cdots\\
      \frac{\partial{f}}{\partial{y_m}}\\
      \end{matrix}
      \right]
      =
      \left[
      \begin{matrix}
      \frac{\partial{y_1}}{\partial{x_1}}&\cdots&\frac{\partial{y_1}}{\partial{x_n}}\\
      \cdots&\cdots&\cdots\\
      \frac{\partial{y_m}}{\partial{x_1}}&\cdots&\frac{\partial{y_m}}{\partial{x_n}}\\
      \end{matrix}
      \right]^T
      \left[
      \begin{matrix}
      \frac{\partial{f}}{\partial{y_1}}\\
      \cdots\\
      \frac{\partial{f}}{\partial{y_m}}\\
      \end{matrix}
      \right]
      \\
      结论：\\
      \nabla_xf = (\frac{\partial{y}}{\partial{x}})^T\nabla_yf\\
      (\frac{\partial{y}}{\partial{x}})^T  雅可比矩阵
   $$
   

  

6. **正向传播时的变换公式：**
   $$
   u^{(l)}=W^{(l)}x^{(l-1)}+b^{(l)}\\
      x^{(l)} = f\big(u^{(l)}\big)\\
   $$
   **计算权重和偏置的梯度：**$\nabla_{W^{(l)}}L = \big(\nabla_{u^{(l)}}L\big)\big(x^{(l-1)}\big)^T\\
      \nabla_{b^{(l)}}L = \nabla_{u^{(l)}}L$

   

   

7. **输出层：**
   $$
      \nabla_{u^{(l)}}L = \big(\nabla_{x^{(l)}}L\big)\odot f^\prime\big(u^{(l)}\big)=\big(x^{(l)}-y\big)\odot f^\prime(u^{(l)})
   $$
   **权重和偏执的梯度可以直接求出：**
   $$
    \nabla_{W^{(l)}}L = \big(x^{(l)}-y\big)\odot f^\prime\big(u^{(l)}\big)\big(x^{(l-1)}\big)^T\\
      \nabla_{b^{(l)}}L = \big(x^{(l)}-y\big)\odot f^\prime\big(u^{(l)}\big)
   $$
   **隐藏层正向传播时的变换：**
   $$
    u^{(l+1)} = W^{(l+1)}x^{(l)} + b^{(l+1)} = W^{(l+1)}f(u^{(l)}) + b^{(l+1)}\\
   $$
   **根据后一层的梯度计算本层的梯度：**
   $$
      \nabla_{u^{(l)}}L = (\nabla_{x^{(l)}}L)\odot{f^\prime(u^{(l)})} = \big((W^{(l+1)})^T\nabla_{u^{(l+1)}}L\big)\odot{f^\prime(u^{(l)})}
   $$
   **定义误差项：**
   $$
    \delta^{(l)} = \nabla_{u^{(l)}}L = 
      \left\{
      \begin{aligned}
      (x^{(l)}-y)\odot{f^\prime(u^{(l)})}\qquad\qquad\quad &l=n_l& 终点\\
      (W^{(l+1)})^T(\delta^{(l+1)})\odot f^\prime(u^{(l)})\qquad&l\neq{n_l}& 递推公式
      \end{aligned}
      \right.
   $$
   


8.  **反向传播算法**

   计算误差项：

   ​		输出层：$\delta^{n_l} = (x^{(n_l)}-y)\odot{f^\prime(u^{(n_l)})}$

   ​		隐含层：$\delta^{l} = (W^{(l+1)})^T\delta^{(l+1)}\odot{f^\prime(u^{(l)})}$

   计算梯度值：

   ​		权重：$\nabla_{W^{(l)}}L = \delta^{(l)}\big(x^{(l-1)}\big)^T$

   ​		偏置：$\nabla_{b^{(l)}}L = \delta^{(l)}$

   梯度下降更新：
   $$
   W^{(l)} = W^{(l)}-\eta\nabla_{W^{(l)}}L\\b^{(l)} = b^{(l)}-\eta\nabla_{b^{(l)}}L
   $$
   