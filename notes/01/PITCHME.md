# Machine Learning 101

---

## Motivation

* **Loan Application Approval System**

* Design a module that predicts whether a borrower (loan applicant) will default on a loan.
* Input application form (name, age, income, etc...)

+++

* Actual Competition

* https://www.kaggle.com/c/GiveMeSomeCredit

* Prizes
    * 1st: $3,000 
    * 2nd: $1,500 
    * 3rd: $500 

+++

* **Solution 1**: Code the rules

* Nested if-then-else?

* Drawback: Updating the rules require another build-test-deploy cycle

+++

* **Solution 2**: Rules Engine

* Rules can be updated without updating the program

* Drawback: Who will write the rules?

+++

* **Solution 3**: Machine Learning

* Now, what is machine learning? What can (and can't) it do?

---

## What is Machine Learning?

* Arthur Samuel (1959): Field of study that gives computers the ability 
to learn without being programmed.

+++

* Arthur Samuel wrote a checkers program that "played thousands of games against itself" to improve.

![Draughts/Checkers](notes/assets/International_draughts.jpg)

---

## What is Machine Learning?

* Tom Mitchell (1998): Study of algorithms that:
    * improve on a given task _T_
    * a certain performance measure _P_
    * given experience _E_

* A well defined learning task is given by _< T, P, E >_

+++

* **Checkers**

* _T_: playing checkers
* _P_: probability of winning
* _E_: simulated games against itself

+++

* **Loan Application System**

* _T_: predicting loan default
* _P_: probability that prediction is correct
* _E_: previous loan application result

---

## What is Machine Learning?

**Typical Programs**

![Data+Program = Output](notes/assets/typical-app.png)

**Machine Learning**

![Data+Output = Program](notes/assets/ml-app.png)

---

## More Examples

**Image recognition**

* https://www.kaggle.com/c/digit-recognizer

![Kaggle Image](https://kaggle2.blob.core.windows.net/competitions/kaggle/3004/logos/front_page.png)

+++

**Predict house price**

* https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![Kaggle Image](https://kaggle2.blob.core.windows.net/competitions/kaggle/5407/media/housesbanner.png)

+++

**Spam Detection**

* https://www.kaggle.com/karthickveerakumar/spam-filter

---

## Types of Problem

* Supervised learning
* Unsupervised learning
* Reinforced learning

---

## Sample ML Algorithms

* Decision Trees
* Artificial Neural Networks
* Support Vector Machine
* Bayesian networks
* Clustering

---

## Machine Learning Workflow

+++

![Image](https://www.ibm.com/developerworks/community/blogs/jfp/resource/BLOGS_UPLOADED_IMAGES/mlworkflowperception.png)

Data > ??? > ML Algorithm > ??? > $$$ 

+++

![Image](https://www.ibm.com/developerworks/community/blogs/jfp/resource/BLOGS_UPLOADED_IMAGES/MLworkflownotsimple.png)

Data > _Data Prep_ > ML Algorithm > _Model_ > _Deploy_ > _Predict_ > $$$ 

---

## Where to start?

* Python + R
  * Python + sklearn
  * Most common

+++

* Mathlab / Octave
  * linear algebra and optimization implementation
  * good for implementing ML algo from scratch (?)

+++

* Java
  * weka
  * spark + mllib
  * deeplearning4j

---
## References

* https://www.coursera.org/learn/machine-learning/
* https://www.kaggle.com/
* http://www.seas.upenn.edu/~cis519/