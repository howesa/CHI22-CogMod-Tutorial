import driver
import driver_agent

import cv2
import time

import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

class driver_env():
    def __init__(self, driver, agent):
        self.driver = driver
        self.agent = agent

        self.total_reward = 0

        self.x_size = 900
        self.y_size = 900

        self.min_x_to_x = 100
        self.max_x_to_x = 500
        self.x_scale = self.max_x_to_x - self.min_x_to_x

        self.y = int(self.y_size/2)

        self.canvas_shape = (self.y_size, self.x_size, 3)
        self.canvas = np.ones(self.canvas_shape) * 1

        self.car_im = cv2.imread("resources/car.png")

    def draw_elements_on_canvas(self):
        self.canvas = np.ones(self.canvas_shape) * 1

        icon = self.car_im
        x = int(self.driver.x * self.x_scale + self.min_x_to_x)
        y = self.y
        self.canvas[y : y + icon.shape[1], x : x + icon.shape[0]] = icon

        text = 'Avg.Score: {}'.format(round(self.total_reward/(self.driver.time+0.01),2))
        self.canvas = cv2.putText(self.canvas, text, (10,20), font, 0.8, (0,0,0), 1, cv2.LINE_AA)

        text = 'Time: {}'.format(round(self.driver.time))
        self.canvas = cv2.putText(self.canvas, text, (10,40), font, 0.8, (0,0,0), 1, cv2.LINE_AA)

    def render(self):
        self.draw_elements_on_canvas()
        cv2.imshow("Game", self.canvas)
        self.key = cv2.waitKey(10)
        return self.canvas

    def simulate(self, until):
        self.driver.reset()
        next_step = 0
        while self.driver.time < until:
            self.render()
            if time.time() > next_step:
                action = self.agent.get_action(self.driver.get_belief())
                self.driver.step(action)
                next_step = time.time() + self.driver.step_time
                self.total_reward += self.driver.reward
            if self.key == 27:
                self.close()
                return
        self.close()

    def close(self):
        cv2.destroyAllWindows()


d = driver.driver(17, continuous = False)
d.learn_t_hat(samples=1000)
a = driver_agent.driver_agent(d)
a.train_agent(debug = True, max_iters = 10)
e = driver_env(d,a)
e.simulate(100)
