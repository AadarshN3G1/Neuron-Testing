
# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

"""
import gurobipy as gp
from gurobipy import GRB

try:
    # Create a new model
    m = gp.Model("mip1")

    # Create variables
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # Set objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")

    print(f"Obj: {m.ObjVal:g}")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")








import gurobipy as gp
from gurobipy import GRB

model = gp.Model("ThresholdTest")

X1 = model.addVar(vtype=GRB.BINARY, name="X1")
X2 = model.addVar(vtype=GRB.BINARY, name="X2")
X3 = model.addVar(vtype=GRB.BINARY, name="X3")
Z = model.addVar(vtype=GRB.BINARY, name="Z")

w1 = 1
w2 = 2
w3 = 3
T = 0

model.addGenConstrIndicator(Z, True, w1*X1 + w2*X2 + w3*X3 >= T, name="Indicator")
model.addGenConstrIndicator(Z, False, w1*X1 + w2*X2 + w3*X3 <= T-1, name="Indicator")


# Set the objective to minimize Z
model.setObjective(Z, GRB.MINIMIZE)

model.optimize()

# Check if the test always passes
if model.objVal == 0:
    print("The threshold test does not always pass for all inputs.")
else:
    print("The threshold test always passes for all inputs.")
"""



"""


import gurobipy as gp
from gurobipy import GRB

w1, w2, w3 = 2, 1, -1  
w1_prime, w2_prime, w3_prime = 1, 2, -1 
T = 1  
T_prime = 0

model = gp.Model("threshold_test_equivalence")

x1 = model.addVar(vtype=GRB.BINARY, name="x1")
x2 = model.addVar(vtype=GRB.BINARY, name="x2")
x3 = model.addVar(vtype=GRB.BINARY, name="x3")

test1_result = model.addVar(vtype=GRB.BINARY, name="test1_result")
test2_result = model.addVar(vtype=GRB.BINARY, name="test2_result")
both_same = model.addVar(vtype=GRB.BINARY, name="both_same")


#model.addConstr((w1 * x1 + w2 * x2 + w3 * x3 >= T) == test1_result, "Test1")
model.addGenConstrIndicator(test1_result, True, w1*x1 + w2*x2 + w3*x3 >= T, name="Indicator1a")
model.addGenConstrIndicator(test1_result, False, w1*x1 + w2*x2 + w3*x3 <= T-1, name="Indicator1b")

#model.addConstr((w1_prime * x1 + w2_prime * x2 + w3_prime * x3 >= T_prime) == test2_result, "Test2")
model.addGenConstrIndicator(test2_result, True, w1_prime*x1 + w2_prime*x2 + w3_prime*x3 >= T_prime, name="Indicator2a")
model.addGenConstrIndicator(test2_result, False, w1_prime*x1 + w2_prime*x2 + w3_prime*x3 <= T_prime-1, name="Indicator2b")

#model.addConstr(test1_result != test2_result, "inequivalence")
model.addGenConstrIndicator(both_same, True, test1_result == test2_result, name="both_same")


model.setObjective(both_same, GRB.MINIMIZE)

model.optimize()

#if model.Status == GRB.OPTIMAL:
#    print("Test 1 and Test 2 are NOT equivalent.")
#    print("Example of non-equivalence: x1=", x1, ", x2=", x2, ", x3=", x3)
#else:
#    print("Test 1 and Test 2 are equivalent for all input settings.")


if model.Status == GRB.INFEASIBLE:
    print("Test 1 and Test 2 are equivalent for all input settings.")
else:
    print("Test 1 and Test 2 are NOT equivalent.")
    print("Example of non-equivalence: x1=", x1.x, ", x2=", x2.x, ", x3=", x3.x)

"""

"""


from gurobipy import Model, GRB, LinExpr

model = Model("test_equivalence")

x1 = model.addVar(vtype=GRB.BINARY, name="x1")
x2 = model.addVar(vtype=GRB.BINARY, name="x2")
x3 = model.addVar(vtype=GRB.BINARY, name="x3")

w1, w2, w3, T = 4, 2, -4, 1
w1_prime, w2_prime, w3_prime, T_prime = 4, 2, -4, 2

test1_expr = LinExpr(w1 * x1 + w2 * x2 + w3 * x3)
test2_expr = LinExpr(w1_prime * x1 + w2_prime * x2 + w3_prime * x3)

#model.addConstr(test1_expr - T >= test2_expr - T_prime, "Test1_covers_Test2")
#model.addConstr(test2_expr - T_prime >= test1_expr - T, "Test2_covers_Test1")
#model.setObjective(0, GRB.MINIMIZE)

f1 = test1_expr - T + .5
f2 = test2_expr - T_prime + .5
#model.addConstr(f1*f2 >= 0)
model.setObjective(f1*f2, GRB.MINIMIZE)

model.optimize()

if model.Status == GRB.OPTIMAL:
    print("The tests are equivalent.")
else:
    print("The tests are not equivalent.")






#using indicator constraints and objective function, test if whenever z1 is passing then z2 is also passing








"""

"""
import gurobipy as gp
from gurobipy import Model,GRB

model = Model("test_equivalence")


x1 = model.addVar(vtype=GRB.BINARY, name="x1")
x2 = model.addVar(vtype=GRB.BINARY, name="x2")
x3 = model.addVar(vtype=GRB.BINARY, name="x3")

w1 = 1  
w2 = 2  
w3 = 3  

T = model.addVar(name="T")  

model.addConstr(w1 * x1 + w2 * x2 + w3 * x3 >= T, "Constraint")

model.setObjective(T, GRB.MAXIMIZE)
model.optimize()
T_plus = model.objVal

model.setObjective(-T, GRB.MAXIMIZE)
model.optimize()
T_minus = -model.objVal

print(f"Largest threshold T+: {T_plus}")
print(f"Smallest threshold T-: {T_minus}")
"""
"""


import gurobipy as gp
from gurobipy import GRB

model = gp.Model("ThresholdTest")

X1 = model.addVar(vtype=GRB.BINARY, name="X1")
X2 = model.addVar(vtype=GRB.BINARY, name="X2")
X3 = model.addVar(vtype=GRB.BINARY, name="X3")

w1 = 2
w2 = 1
w3 = -1
T = 1

w1_prime = 1
w2_prime = 2
w3_prime = -1
T_prime = 0

Z1 = model.addVar(vtype=GRB.BINARY, name="Z1")
Z2 = model.addVar(vtype=GRB.BINARY, name="Z2")
C = model.addVar(vtype=GRB.BINARY, name="C")  


model.addConstr(Z1 == 1, name="Test1_True")  
model.addGenConstrIndicator(Z1, True, w1 * X1 + w2 * X2 + w3 * X3 >= T, name="Indicator1_True")
model.addGenConstrIndicator(Z1, False, w1 * X1 + w2 * X2 + w3 * X3 <= T - 1, name="Indicator1_False")


model.addConstr(Z2 == 1, name="Test2_True")  
model.addGenConstrIndicator(Z2, True, w1_prime * X1 + w2_prime * X2 + w3_prime * X3 >= T_prime, name="Indicator2_True")
model.addGenConstrIndicator(Z2, False, w1_prime * X1 + w2_prime * X2 + w3_prime * X3 <= T_prime - 1, name="Indicator2_False")


model.addConstr(C >= Z1 + Z2 - 1, name="Inequivalence1")
model.addConstr(C <= Z1 + Z2, name="Inequivalence2")     

model.setObjective(C, GRB.MINIMIZE)

model.optimize()

if model.status == GRB.OPTIMAL:
    if model.objVal == 0:
        print("The two threshold tests are equivalent.")
    else:
        print("The two threshold tests are not equivalent.")
else:
    print("Optimization was not successful.")

"""
"""
import gurobipy as gp
from gurobipy import GRB

model = gp.Model("ThresholdTestEquivalence")

X1 = model.addVar(vtype=GRB.BINARY, name="X1")
X2 = model.addVar(vtype=GRB.BINARY, name="X2")
X3 = model.addVar(vtype=GRB.BINARY, name="X3")
Z1 = model.addVar(vtype=GRB.BINARY, name="Z1")
Z2 = model.addVar(vtype=GRB.BINARY, name="Z2")

w1, w2, w3, T = 2, 2, -2, 0  
wp1, wp2, wp3, Tp = 2, 2, -2, -1

model.addGenConstrIndicator(Z1, True, w1 * X1 + w2 * X2 + w3 * X3 >= T, name="Indicator1_True")
model.addGenConstrIndicator(Z1, False, w1 * X1 + w2 * X2 + w3 * X3 <= T - 1, name="Indicator1_False")

model.addGenConstrIndicator(Z2, True, wp1 * X1 + wp2 * X2 + wp3 * X3 >= Tp, name="Indicator2_True")
model.addGenConstrIndicator(Z2, False, wp1 * X1 + wp2 * X2 + wp3 * X3 <= Tp - 1, name="Indicator2_False")



model.addConstr(Z1 + Z2 == 1, name="Inequivalence")

model.setObjective(0, GRB.MINIMIZE)

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Threshold tests are not equivalent")
    for v in model.getVars():
        print(f"{v.VarName} = {v.X}")

else:
    print("Threshold tests are equivalent")

  
#given two thresold tests (they are eqivalent), what is the maximum(optimize the thresold)/minimum we can change the thresolds while keeping the same behavior 


#M = 1000
#model.addConstr(w1 * X1 + w2 * X2 + w3 * X3 - T <= M * (1 - Z1), name="Z1_ge")
#model.addConstr(w1 * X1 + w2 * X2 + w3 * X3 - T >= Z1, name="Z1_le")
#model.addConstr(wp1 * X1 + wp2 * X2 + wp3 * X3 - Tp <= M * (1 - Z2), name="Z2_ge")
#model.addConstr(wp1 * X1 + wp2 * X2 + wp3 * X3 - Tp >= Z2, name="Z2_le")
#a strict condition must be met in order to define z1 as passing, hence z1 = 1. 

"""
"""
import gurobipy as gp
from gurobipy import GRB

model = gp.Model("ThresholdOptimization")

X1 = model.addVar(vtype=GRB.BINARY, name="X1")
X2 = model.addVar(vtype=GRB.BINARY, name="X2")
X3 = model.addVar(vtype=GRB.BINARY, name="X3")

Z1 = model.addVar(vtype=GRB.BINARY, name="Z1")
Z2 = model.addVar(vtype=GRB.BINARY, name="Z2")

w1, w2, w3 = 2, 2, -2
wp1, wp2, wp3 = 2, 2, -2

#T = model.addVar(vtype=GRB.CONTINUOUS, name="T")
T = 1
Tp = model.addVar(vtype=GRB.INTEGER, lb=T, name="Tp")

model.addGenConstrIndicator(Z1, True, w1 * X1 + w2 * X2 + w3 * X3 >= T, name="Indicator1_True")
model.addGenConstrIndicator(Z1, False, w1 * X1 + w2 * X2 + w3 * X3 <= T - 1, name="Indicator1_False")

model.addGenConstrIndicator(Z2, True, wp1 * X1 + wp2 * X2 + wp3 * X3 >= Tp, name="Indicator2_True")
model.addGenConstrIndicator(Z2, False, wp1 * X1 + wp2 * X2 + wp3 * X3 <= Tp - 1, name="Indicator2_False")

#model.addConstr(Z1 == Z2, name="Equivalence")
model.addConstr(Z1 + Z2 == 1, name="Inequivalence")



model.setObjective(Tp, GRB.MINIMIZE)


model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimized thresholds:")
    for v in model.getVars():
        print(f"{v.VarName} = {v.X}")
else:
    print("No optimal solution found")
"""
"""

import gurobipy as gp
from gurobipy import GRB

model = gp.Model("ThresholdOptimization")

X1 = model.addVar(vtype=GRB.BINARY, name="X1")
X2 = model.addVar(vtype=GRB.BINARY, name="X2")
X3 = model.addVar(vtype=GRB.BINARY, name="X3")

Z1 = model.addVar(vtype=GRB.BINARY, name="Z1")
Z2 = model.addVar(vtype=GRB.BINARY, name="Z2")

w1, w2, w3 = 2, 2, -2
wp1, wp2, wp3 = 2, 2, -2

T = 1
Tp = model.addVar(vtype=GRB.INTEGER, lb=T, name="Tp")

model.addGenConstrIndicator(Z1, True, w1 * X1 + w2 * X2 + w3 * X3 >= T, name="Indicator1_True")
model.addGenConstrIndicator(Z1, False, w1 * X1 + w2 * X2 + w3 * X3 <= T - 1, name="Indicator1_False")

model.addGenConstrIndicator(Z2, True, wp1 * X1 + wp2 * X2 + wp3 * X3 >= Tp, name="Indicator2_True")
model.addGenConstrIndicator(Z2, False, wp1 * X1 + wp2 * X2 + wp3 * X3 <= Tp - 1, name="Indicator2_False")

model.addConstr(Z1 + Z2 == 1, name="Inequivalence")

model.setObjective(Tp, GRB.MAXIMIZE)

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimized thresholds:")
    for v in model.getVars():
        print(f"{v.VarName} = {v.X}")
else:
    print("No optimal solution found")



"""

"""

import gurobipy as gp
from gurobipy import GRB

model = gp.Model("ThresholdOptimization")

X1 = model.addVar(vtype=GRB.BINARY, name="X1")
X2 = model.addVar(vtype=GRB.BINARY, name="X2")
X3 = model.addVar(vtype=GRB.BINARY, name="X3")
Z1 = model.addVar(vtype=GRB.BINARY, name="Z1")
Z2 = model.addVar(vtype=GRB.BINARY, name="Z2")

T = 16
Tp = model.addVar(vtype=GRB.INTEGER, ub=T,lb=float("-inf"), name="Tp")

w1, w2, w3 = 20, 20, -20
wp1, wp2, wp3 = 20, 20, -20

model.addGenConstrIndicator(Z1, True, w1 * X1 + w2 * X2 + w3 * X3 >= T, name="Indicator1_True")
model.addGenConstrIndicator(Z1, False, w1 * X1 + w2 * X2 + w3 * X3 <= T - 1, name="Indicator1_False")

model.addGenConstrIndicator(Z2, True, wp1 * X1 + wp2 * X2 + wp3 * X3 >= Tp, name="Indicator2_True")
model.addGenConstrIndicator(Z2, False, wp1 * X1 + wp2 * X2 + wp3 * X3 <= Tp - 1, name="Indicator2_False")


model.addConstr(Z1 + Z2 == 1)

model.setObjective(Tp, GRB.MAXIMIZE)

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimized thresholds:")
    for v in model.getVars():
        print(f"{v.VarName} = {v.X}")
else:
    print("No optimal solution found")



#t' has to greater than equal to T (upper bound)


"""
    
"""
from gurobipy import Model, GRB, quicksum
import random

class ThresholdModel:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def always_passes(self):

        model = Model("always_passes")

        x = model.addVars(len(self.weights), vtype=GRB.BINARY, name="x")

        #model.addConstr(quicksum(self.weights[i] * x[i] for i in range(len(self.weights))) <= self.threshold - 1,
        #                "threshold_constraint")
        model.addConstr(quicksum(weight * x[i] for weight,i in zip(self.weights,x)) <= self.threshold - 1,
                        "threshold_constraint")

        model.setObjective(0, GRB.MINIMIZE)

        model.optimize()

        #for v in model.getVars():
        #    print(f"{v.VarName} = {v.X}")

        #return model.status == GRB.OPTIMAL
        return model.status == GRB.INFEASIBLE

    def always_fails(self):
        pass

    def equivalent_to(self,other_test):
        assert len(self.weights) == len(other_test.weights)
        pass

if __name__ == "__main__":
    #weights = [3, 4, 11, 1, -4, -3, 3, 5, 2, 1]
    weights = [ random.randint(-10,10) for _ in range(5) ]
    #threshold = sum(weight for weight in weights if weight < 0)
    threshold = 0

    model = ThresholdModel(weights, threshold)

    result = model.always_passes()
    print("Always passes:", result)

    result = model.equivalent_to(model)
"""


import random
from gurobipy import Model, GRB, quicksum

class ThresholdModel:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def equivalent_to(self, other_test):
        assert len(self.weights) == len(other_test.weights), "Weights must be of the same length."

        model = Model("equivalence_test")
        #model.setParam('OutputFlag', 1)

        x1 = model.addVars(len(self.weights), vtype=GRB.BINARY, name="x1")
        #x2 = model.addVars(len(self.weights), vtype=GRB.BINARY, name="x2")

        Z1 = model.addVar(vtype=GRB.BINARY, name="Z1")
        Z2 = model.addVar(vtype=GRB.BINARY, name="Z2")

        model.addGenConstrIndicator(Z1, True, quicksum(self.weights[i] * x1[i] for i in range(len(self.weights))) >= self.threshold, name="Z1_Pass1")
        model.addGenConstrIndicator(Z1, False, quicksum(self.weights[i] * x1[i] for i in range(len(self.weights))) <= self.threshold-1, name="Z1_Fail2")
        model.addGenConstrIndicator(Z2, False, quicksum(other_test.weights[i] * x1[i] for i in range(len(other_test.weights))) <= other_test.threshold-1, name="Z2_Fail1")
        model.addGenConstrIndicator(Z2, True, quicksum(other_test.weights[i] * x1[i] for i in range(len(other_test.weights))) >= other_test.threshold, name="Z2_Pass2")

        model.addConstr(Z1 + Z2 == 1, "Inequivalence")

        model.setObjective(0, GRB.MINIMIZE)

        model.optimize()

        return model.status != GRB.OPTIMAL 

if __name__ == "__main__":
    weights = [random.randint(1000000,1010000) for _ in range(1000)]
    #threshold = 0 # random.randint(-10, 10)
    # random 
    threshold = int(sum(weights)/2)

    print("Weights:", weights)
    print("Threshold:", threshold)

    model = ThresholdModel(weights, threshold)
    other_model = ThresholdModel(weights, threshold)

    result_equivalence = model.equivalent_to(other_model)
    print("Equivalent thresholds:", result_equivalence)


