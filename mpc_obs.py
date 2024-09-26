import time, sys

from casadi import *
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
# from utils import plot_pendulum
import numpy as np
import scipy.linalg
import rospy
import math
import random

class mpc(object):
    def __init__(self):
        self.RUN_ONCE = True
        self.tcomp_sum = 0
        self.tcomp_max = 0
        self.acados_solver = 0
        self.N = 0
        self.Tf = 0
        self.active = False
        

    def kine_model(self):

        model = types.SimpleNamespace()
        model_name = 'kine_ode'

        x = SX.sym('x')
        y = SX.sym('y')
        phi = SX.sym('phi')
        states = vertcat(x, y, phi)
               
        # controls
        v = SX.sym('v')
        phidot = SX.sym('phidot')
        u = vertcat(v,phidot)
        
        # statesdot
        xdot  = SX.sym('xdot')
        ydot  = SX.sym('ydot')
        phid = SX.sym('phid')
        
        statesdot = vertcat(xdot, ydot, phid)
        
        obs_x = SX.sym('obs_x')
        obs_y = SX.sym('obs_y')
        dist =  1/exp(((x- obs_x )/0.17)**2 + ((y- obs_y )/0.17)**2)

        # parameters
        p = vertcat(obs_x, obs_y)
        
        # dynamics
        f_expl = vertcat(v*cos(phi),
                        v*sin(phi),
                        phidot
                        )


        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = states
        model.xdot = statesdot
        model.u = u
        model.p = p
        model.name = model_name
        
        model.cost_y_expr = vertcat(states, u, dist) 
        model.cost_y_expr_e = vertcat(states, dist)    

        return model

    ##############################################    
    def acado_set(self, x0):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        self.Tf = 6
        self.N = 30

        # set model
        model = self.kine_model()

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu + 1
        ny_e = nx + 1
        npa = model.p.size()[0]


        ocp.model = model
        # set dimensions
        ocp.dims.N = self.N
        ocp.dims.np = npa

        # set cost module
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
          
        ocp.parameter_values = np.array([100, 100])

        Q = np.diag([0.1, 0.1, 0.05])
        R = np.diag([0.001, 0.01])
        w_obst = 25
        Qe = np.diag([1, 1, 0.1])

        ocp.cost.W = scipy.linalg.block_diag(Q, R, w_obst)

        ocp.cost.W_e = scipy.linalg.block_diag(Qe, w_obst) 

        ocp.cost.yref  = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))
        
        # set constraints
        vmax = 0.5
        phidotmax = 0.785
        # ocp.constraints.constr_type = 'BGH'
        
        ocp.constraints.idxbu = np.array([0, 1]) 
        ocp.constraints.lbu = np.array([-vmax , -phidotmax])
        ocp.constraints.ubu = np.array([vmax, phidotmax])
        
        ocp.constraints.idxbx = np.array([0, 1, 2])
        ocp.constraints.lbx = np.array([-3, -3, -2*math.pi])
        ocp.constraints.ubx = np.array([3, 3, 2*math.pi])

        ocp.constraints.x0 = x0
        
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' 
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP' 
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp.solver_options.qp_solver_iter_max = 500
        ocp.solver_options.nlp_solver_max_iter = 1000  
        

        ocp.solver_options.tf = self.Tf

        r = random.randint(1,100)

        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + str(r)+ model.name + '.json')
        
        return  acados_ocp_solver
    ############################################

    def run_mpc(self, ydes, x0, obs_x, obs_y):

        ## Assuming ydes is single point 

        if self.RUN_ONCE:
            self.acados_solver = self.acado_set(x0)
            self.RUN_ONCE = False
           

        for j in range(self.N):
            yref = np.append(ydes,0)
            self.acados_solver.set(j, "yref", yref)
            self.acados_solver.set(j,"p", np.array([obs_x, obs_y]))
            self.acados_solver.set(j,"x", x0)
            self.acados_solver.set(j,"u", np.zeros(2))
            
                     
        yref_N = np.append(ydes[0:3], 0)
        self.acados_solver.set(self.N, "yref", yref_N)
        self.acados_solver.set(self.N, "p", np.array([obs_x, obs_y]) )
        self.acados_solver.set(self.N,"x", x0)
        
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)
        # solve ocp
        t = time.time()

        status = self.acados_solver.solve()
        
        if status != 0:
            rospy.loginfo("acados returned status {}".format(status))
            
        elapsed = time.time() - t

        
        # manage timings
        self.tcomp_sum += elapsed
        if elapsed > self.tcomp_max:
            tcomp_max = elapsed

        # get solution
        u0 = self.acados_solver.get(0, "u")
        

        return u0
