## This is a general markup file for equations
# may require adjustments

# Gradient update  
\frac{\partial}{\partial\texttheta_j}MSE(\texttheta) = 
\frac{2}{m}\sum{i=1}^m\left(\texttheta^T\cdot\mathbf x^{(i)}-y^{(i)}\right)
x^{(i)}_j
\nabla_\texttheta MSE(\texttheta) = 
\left\lgroup 

# Gradient vector of cost function
\matrix{ \frac{\partial}{\partial\texttheta_0}MSE(\texttheta)
\cr
\frac{\partial}{\partial\texttheta_1}MSE(\texttheta)
\cr \vdots \cr
\frac{\partial}{\partial\texttheta_n}MSE(\texttheta) 
\right\rgroup = \frac{2}{m}\mathbfX^T\cdot
(\mathbfX\cdot\texttheta-\mathbfy)

# Gradient Descent step
\theta^{next step}=\texttheta-\eta\nabla_\textthetaMSE(\texttheta)

# Regularization term
\alpha\sum_{i=1}^n\texttheta_i^2

# Ridge Regression cost function
J(\theta) + \alpha\frac{1}{2}\sum_{i=1}^n\theta^2_i

# L2 norm of the weight vector 
\frac{1}{2}(||\mathbfw||_2)^2

# As the normalization term
\alpha\mathbfw

# Ridge Regression closed-form solution
\hat\theta = \left(\mathbfX^T\cdot\mathbfX + \alpha\mathbfW\right)^{-1}\cdoty

# Lasso Regression cost function using \ell_1 norm
J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^n \vert \theta_i \vert

# Elastic Net 
J( \theta ) = MSE( \theta ) + 
r \alpha \sum_{i=1}^n \vert \theta_i \vert +
\frac{1 - r}{2} \alpha \sum_{i=1}^n \theta_i^2

# Logistic Regression vectorized estimated probability
\hat p = h_\theta(\mathbf x) = \sigma \left( \theta^T \cdot \mathbf x \right))

# Logistic Function
\sigma(t) = \frac{1}{1 + exp(-t)}

# Logistic Regression model prediction
\haty = \left\{\eqalign{
  0 if \hat p < 0.5,\\
  1 if \hat p \geq 0.5
}

# Softmax score for class k
s_k \left( \mathbf x) \right) = \left( \theta^{(k)} \right)^T \cdot \mathbf x

# Softmax function
\hat p_k = \sigma(\mathbfs(\mathbf(x)))_k = 
\frac{exp(s_k(\mathbf(x)))}
{\sum_{j=1}^K exp \left( s_j \left( \mathbf x \right) \right)}

# Softmax Regression classifier 
\haty = \underset{\rm k}{\rm argmax} s_k \left( \mathbf x \right) = 
\underset{\rm k}{\rm argmax} \left( \left( \theta^{(k)} \right)^T \cdot \mathbf x \right)

# Cross entropy cost function
J(\Theta) = - 1/m \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} log \left( \hat p_k^{(i)}  \right)

# Softmax Regression classifier prediction
\hat y =  \underset {\rm k}{\rm argmax} \sigma \left( \mathbf s ( \mathbf x )_k  \right) =
underset{\rm k}{\rm argmax} s_k \left( \mathbf x  \right) = 
\underset{\rm k}{\rm argmax} \left( \left( \theta^{(k)}  \right)^T \cdot \mathbf x \right)

# Cross entropy cost function
J(\Theta) = - 1/m \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} log \left( \hat p_k^{(i)} \right)

# Cross entropy gradient vector for class K
\nabla_\theta^{(k)} J(\Theta) = 1/m 
\sum_{i=1}^m \left( \hat p_k^{(i)} y_k^{(i)} \right) \mathbf x^{(i)}
