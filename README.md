# Labyrinth

World model
Exploration based on parameter uncertainty
No duplicate data

Evaluation: Search world model for best solution, take one real step, repeat


How to find where we are uncertain (further away)? Need a model for proposing valid/possible states?
Value function but for uncertainty? In what direction are we most uncertain?


Idea: Predict probability of switching instead? How? Possibly add probability of changing as filter?

What states are valid and possible to reach?
What is the value of each?
Search backwards?

Measure uncertainty (KL or other)
Value function for uncertainty
Loop back to start position (value function for start position) if game over

Use memory if we have observed something before

Adding dropout or noise to a normal network would also give a measure of the uncertainty? Could work 
very well when the problem is deterministic?

Idea: If we do distance to memory we have an easier time knowing if this is far from something we have seen before?

VI might be fitting to our observed but also overfitting what we have not seen...

How do we get someplace? We can use our world model? How? We can backprop to find the actions we need to take to get from A to B in N steps. We can build a model that predicts the minimum number of steps between two states?

We can remember how go get to each state that we have seen?

Note: After we have created a dream world we now have accesss to a differentiable world model(!) Only seems to work with RelaxedOneHotCategorical

Q-learning to find best path?
