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



