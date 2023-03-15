import math
import time
import carla
import cv2
import numpy as np
import random

class CarlaEnv:

    STEER_AMT = 1.0

    im_width = 640
    im_height = 480


    def __init__(self, host="localhost"):

        self.port = random.randint(2000, 3000)

        print(self.port)

        self.client = carla.Client(host, 2000)
        self.client.set_timeout(5.0)

        self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]

        self.speed_limit = 40

        self.reward_total = 0
        self.prev_dist = 0

        self.actor_list = None
        self.collision_hist = None

        self.n_actions = 3

        self.state = []
        self.action_space = [0,1,2]
        self.episode_start = None


    def reset(self):

        #clean actor_list
        if self.actor_list is not None:
            self.actors_destroy()

        # claan collision_hist
        if self.collision_hist is not None:
            self.collisions_destroy()

        self.collision_hist = []
        self.actor_list = []

        self.vehicle = self.world.spawn_actor(self.vehicle_bp, random.choice(self.world.get_map().get_spawn_points()))
        self.actor_list.append(self.vehicle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.vehicle.set_simulate_physics(True)

        time.sleep(4)  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        camera_bp = self.blueprint_library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', f'{self.im_width}')
        camera_bp.set_attribute('image_size_y', f'{self.im_height}')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(self._camera_callback)


        collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        collision_sensor_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, collision_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))

        #verifica se a camera já ativou o callback
        while len(self.get_state()) == 0:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.get_state()

    def _camera_callback(self, image):

        """
        In the _camera_callback function,
        we first convert the raw data of the carla.Image object to
        a Numpy array using np.frombuffer.
        Then we reshape the array to match the dimensions
        of the image (height, width, channels),
        and remove the alpha channel if it exists.
        Finally, we return the resulting Numpy array.
        Note that the callback function will be called
        every time the camera sensor captures an image."
        """

        # Convert carla.Image to Numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        self.state = array

    def collision_data(self, event):
        self.collision_hist.append(event)

    def get_state(self):
        if len(self.state)!=0:
            cv2.imwrite('output/CarlaEnv-640x480.jpg', self.state)
        return self.state

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 20:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        self.reward_total += reward

        return self.get_state(), reward, done, None

    def close(self):

        if self.actor_list is not None:
            for actor in self.actor_list:
                actor.destroy()

        if self.world is not None:
            self.world = None
        if self.client is not None:
            self.client = None

    def actors_destroy(self):
        if self.actor_list is not None:

            #self.collision_sensor.unlisten()
            if self.collision_sensor and self.collision_sensor.is_alive:
                self.actor_list.remove(self.collision_sensor)
                self.collision_sensor.destroy()
                self.collision_sensor = None


            if self.camera and self.camera.is_alive:
                self.camera.stop()
                self.actor_list.remove(self.camera)
                time.sleep(1)
                self.camera.destroy()
                self.camera = None


            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                self.actor_list.remove(self.vehicle)
                self.vehicle.destroy()
                self.vehicle = None




            #self.actor_list.clear()

    def collisions_destroy(self):
        if self.collision_hist is not None:
            self.collision_hist.clear()
