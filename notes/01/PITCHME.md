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

![Image](assets/md/assets/International_draughts.jpg)

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

![Image](assets/md/assets/typical-app.png)

**Machine Learning**

![Image](assets/md/assets/ml-app.png)

---

## Types of Learning Algorithms

+++

## Supervised

+++

## Unsuppervised

---
## References

* https://www.coursera.org/learn/machine-learning/
* https://www.kaggle.com/
* http://www.seas.upenn.edu/~cis519/