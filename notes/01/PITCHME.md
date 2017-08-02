# Machine Learning 101

---

## Motivation

* **Loan Application Approval System**

* Design a module that predicts whether a borrower (loan applicant) will default on a loan.
* Input application form (name, age, income, etc...)

+++

* This was an actual competition in kaggle

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

* Now, what is machine learning?

---

## What is Machine Learning?

* Arthur Samuel (1959): Field of study that gives computers the ability 
to learn without being programmed.

+++

* Arthur Samuel wrote a checkers program that "played thousands of games against itself"[2] to improve.

![Draughts/Checkers](notes/assets/International_draughts.jpg)

---

## What is Machine Learning?

* Tom Mitchell (1998): Study of algorithms that:[2][3]
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

## Types of Learning Algorithms

* Supervised learning
* Unsupervised learning

+++

**Supervised learning**

* In supervised learning, we are given a data set and already know what our correct output should look like
* The model, given new inputs, will give a _predicted_ output

+++

**Supervised learning**

Examples:
* Given a picture of a person, we have to predict their age on the basis of the given picture (Regression)
* Given a patient with a tumor, we have to predict whether the tumor is malignant or benign (Classification)

+++

**Unsupervised learning**

* Unsupervised learning allows us to approach problems with little or no idea what our results should look like.
* We can derive structure from data where we don't necessarily know the effect of the variables.
+++

**Unsupervised learning**

Example:
* Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

---

## Machine Learning Workflow

![Image](https://www.ibm.com/developerworks/community/blogs/jfp/resource/BLOGS_UPLOADED_IMAGES/mlworkflowperception.png)

Data > ??? > ML Algorithm > ??? > $$$ [4]

+++

![Image](https://www.ibm.com/developerworks/community/blogs/jfp/resource/BLOGS_UPLOADED_IMAGES/MLworkflownotsimple.png)

Data > _Data Prep_ > ML Algorithm > _Model_ > _Deploy_ > _Predict_ > $$$  [4]

+++

### Our focus will be on

* Example ML libraries to generate models, and make predictions
* Look at a couple of sample algorithms, and how they work
* What to do when predictions are incorrect (bias-vs-variance)

---

## References

* [1] https://www.kaggle.com/
* [2] https://www.coursera.org/learn/machine-learning/
* [3] http://www.seas.upenn.edu/~cis519/
* [4] https://en.wikipedia.org/wiki/Machine_learning
* [5] https://www.ibm.com/developerworks/community/blogs/jfp/entry/The_Machine_Learning_Workflow?lang=en

---

## More Sample Problems

**Image recognition**

* https://www.kaggle.com/c/digit-recognizer

* In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images.

![Kaggle Image](https://kaggle2.blob.core.windows.net/competitions/kaggle/3004/logos/front_page.png)


+++

**Predict house price**

* https://www.kaggle.com/c/house-prices-advanced-regression-techniques

* With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

![Kaggle Image](https://kaggle2.blob.core.windows.net/competitions/kaggle/5407/media/housesbanner.png)

+++

**Fake News Detection**

* https://www.kaggle.com/mrisdal/fake-news

* The latest hot topic in the news is fake news and many are wondering what data scientists can do to detect it and stymie its viral spread. T

+++

**Climate Analysis**

* https://www.kaggle.com/cwiloc/climate-data-from-ocean-ships

* Sample data from 18th to early 19th century sailors. Find any sort of correlation.
