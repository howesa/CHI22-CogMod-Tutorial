import random
import numpy as np


class physics_env():

    def __init__(self, veh1_straight_pos = [-1.825, -10], veh2_turn_pos = [3.65, 1.825], veh1_straight_vel = 8.94, veh2_turn_vel = 0.0 ,sim_freq = 0.05):
        self.veh1_straight_pos = np.array(veh1_straight_pos)
        self.veh1_straight_start_y_range = [-80,-2]
        self.veh2_turn_pos = np.array(veh2_turn_pos)
        self.veh1_straight_vel = veh1_straight_vel
        self.veh1_straight_vel_range = [5,20]
        self.veh2_turn_vel = veh2_turn_vel
        self.init_veh1_pos = np.array(veh1_straight_pos)
        self.init_veh2_pos = np.array(veh2_turn_pos)
        self.init_veh1_vel = veh1_straight_vel
        self.init_veh2_vel = veh2_turn_vel
        self.sim_freq = sim_freq
        self.time = 0
        self.save_trace = False

    def reset(self, y_start = None):
        self.veh1_straight_pos = self.init_veh1_pos
        self.veh1_straight_pos[1] = np.random.uniform(self.veh1_straight_start_y_range[0],
                                                      self.veh1_straight_start_y_range[1])
        if y_start:
            self.veh1_straight_pos[1] = y_start
        self.veh2_turn_pos = self.init_veh2_pos
        # self.veh1_straight_vel = np.random.uniform(self.veh1_straight_vel_range[0],
        #                                            self.veh1_straight_vel_range[1])
        self.y_start = self.veh1_straight_pos[1] # for logging
        self.veh2_turn_vel = self.init_veh2_vel
        self.trace = []
        self.time = 0

    def get_random_b_spawn(self):
        return

    def get_state(self):
        vel = self.veh1_straight_vel
        # acc = self.actor_b.get_acceleration().x
        dist = self.get_distance()

        state = {}

        state['distance'] = dist
        state['speed'] = vel
        state['acceleration'] = 0

        return state

    def get_distance(self):
        return np.linalg.norm(self.veh1_straight_pos-self.veh2_turn_pos)

    def detect_collision(self):
        dist = np.fabs(self.veh1_straight_pos - self.veh2_turn_pos)
        return dist[1]<3.58 and dist[0]<1.645

    def simulate_go(self):
        done = False
        collision = False
        steps = 0
        self.veh2_turn_vel = 8.94
        # self.actor_a.enable_constant_velocity(carla.Vector3D(5,0.0,0.0))
        while not done:
            steps += 1
            self.tick()
            # dist = np.linalg.norm(self.veh1_straight_pos-self.veh2_turn_pos)
            #print(dist)

            if self.detect_collision():
                done = True
                collision = True
                # print(dist)
            # loc = self.actor_a.get_location()
            #print(loc.x, loc.y)
            # if self.veh2_turn_pos[0] <= 0 and self.veh2_turn_pos[1] <=-1.7: #old condition with loose constraint on y-axis position
            if self.veh2_turn_pos[0] <= -3.65:
                #print(self.veh2_turn_pos[0],self.veh2_turn_pos[1])
                done = True
            # Stepped for too long: something wrong with the env?
            if steps > 1000:
                done = True

        return collision, steps

    def tick(self):
        self.time += self.sim_freq
        veh1_current_pos = self.veh1_straight_pos
        veh2_current_pos = self.veh2_turn_pos

        veh1_dist_travel = self.veh1_straight_vel*self.sim_freq
        veh2_dist_travel = self.veh2_turn_vel * self.sim_freq

        veh2_turn_radius = 13.301859
        veh2_turn_centre_x = -5.018750
        veh2_turn_centre_y = 11.406250

        self.veh1_straight_pos = self.veh1_straight_pos + np.array([0.0, veh1_dist_travel])
        angle = np.arctan2(self.veh2_turn_pos[1]-veh2_turn_centre_y,
                           self.veh2_turn_pos[0] - veh2_turn_centre_x)
        angle_increment = veh2_dist_travel/veh2_turn_radius
        angle -= angle_increment
        self.veh2_turn_pos = np.array([veh2_turn_radius*np.cos(angle)+veh2_turn_centre_x,
                                       veh2_turn_radius*np.sin(angle)+veh2_turn_centre_y])

        if self.save_trace:
            self.trace.append([self.time, veh1_current_pos, veh2_current_pos, self.get_distance(), self.detect_collision()])
