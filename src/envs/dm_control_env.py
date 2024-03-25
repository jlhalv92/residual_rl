import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
   # from dm_control import suite
    from dm_control.suite.wrappers import pixels

from src.dm_control import suite
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
from ..utils.ultils_viewer import CV2Viewer

class DMControl(Environment):
    """
    Interface for dm_control suite Mujoco environments. It makes it possible to
    use every dm_control suite Mujoco environment just providing the necessary
    information.

    """
    def __init__(self, domain_name, task_name, horizon=None, gamma=0.99, task_kwargs=None,
                 dt=.01, width_screen=480, height_screen=480, camera_id=0,
                 use_pixels=False, pixels_width=64, pixels_height=64):
        """
        Constructor.

        Args:
             domain_name (str): name of the environment;
             task_name (str): name of the task of the environment;
             horizon (int): the horizon;
             gamma (float): the discount factor;
             task_kwargs (dict, None): parameters of the task;
             dt (float, .01): duration of a control step;
             width_screen (int, 480): width of the screen;
             height_screen (int, 480): height of the screen;
             camera_id (int, 0): position of camera to render the environment;
             use_pixels (bool, False): if True, pixel observations are used
                rather than the state vector;
             pixels_width (int, 64): width of the pixel observation;
             pixels_height (int, 64): height of the pixel observation;

        """
        # MDP creation
        self.env = suite.load(domain_name, task_name, task_kwargs=task_kwargs)
        random_rgb = np.random.rand(3)
        self.env.physics.named.model.geom_rgba['torso', :3] = random_rgb
        print("torso mass: ", self.env.physics.named.model.body_mass['torso'])
        # self.env.physics.named.model.geom_rgba['left_foot', :3] = random_rgb
        # self.env.physics.named.model.body_mass['torso'] = self.env.physics.named.model.body_mass['torso'].copy()*2
        # # self.env.physics.named.model.body_mass['left_foot'] = 40.
        # # self.env.physics.named.model.geom_size["right_thigh"] = [0.05,  0.065, 0.]
        print(self.env.physics.named.model.geom_friction['left_foot'])
        # self.env.physics.named.model.geom_friction['left_foot'] = 3*[.001]
        # self.env.physics.named.model.geom_friction['right_foot'] =3*[.001]

        # self.env.physics.named.model.geom_friction['floor'] =3*[.001]



        # print(self.env.physics.named.model.body_mass['right_foot'])


        # res = self.env.physics.model.hfield_nrow[_HEIGHTFIELD_ID]
        # assert res == self.env.physics.model.hfield_ncol[_HEIGHTFIELD_ID]
        # # Sinusoidal bowl shape.
        # row_grid, col_grid = np.ogrid[-1:1:res * 1j, -1:1:res * 1j]
        # radius = np.clip(np.sqrt(col_grid ** 2 + row_grid ** 2), .04, 1)
        # bowl_shape = .5 - np.cos(2 * np.pi * radius) / 2
        # # Random smooth bumps.
        # terrain_size = 2 * self.env.physics.model.hfield_size[_HEIGHTFIELD_ID, 0]
        # bump_res = int(terrain_size / self.env._TERRAIN_BUMP_SCALE)
        # bumps = self.random.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
        # smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
        # # Terrain is elementwise product.
        # terrain = bowl_shape * smooth_bumps
        # start_idx = self.env.physics.model.hfield_adr[_HEIGHTFIELD_ID]
        # self.env.physics.model.hfield_data[start_idx:start_idx + res ** 2] = terrain.ravel()
        # self.env.super().initialize_episode(self.env.physics)
        #
        # # If we have a rendering context, we need to re-upload the modified
        # # heightfield data.
        # if self.env.physics.contexts:
        #     with self.env.physics.contexts.gl.make_current() as ctx:
        #         ctx.call(mjlib.mjr_uploadHField,
        #                  physics.model.ptr,
        #                  physics.contexts.mujoco.ptr,
        #                  _HEIGHTFIELD_ID)


        if use_pixels:
            self.env = pixels.Wrapper(self.env, render_kwargs={'width': pixels_width, 'height': pixels_height})

        # get the default horizon
        if horizon is None:
            horizon = self.env._step_limit

        # Hack to ignore dm_control time limit.
        self.env._step_limit = np.inf

        if use_pixels:
            self._convert_observation_space = self._convert_observation_space_pixels
            self._convert_observation = self._convert_observation_pixels
        else:
            self._convert_observation_space = self._convert_observation_space_vector
            self._convert_observation = self._convert_observation_vector

        # MDP properties
        action_space = self._convert_action_space(self.env.action_spec())
        observation_space = self._convert_observation_space(self.env.observation_spec())
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        self._height_screen = height_screen
        self._width_screen = width_screen
        self._viewer = CV2Viewer("dm_control", dt, self._width_screen, self._height_screen)
        self._camera_id = camera_id

        super().__init__(mdp_info)

        self._state = None

    def reset(self, state=None):
        if state is None:
            self._state = self._convert_observation(self.env.reset().observation)
        else:
            raise NotImplementedError

        return self._state

    def step(self, action):
        step = self.env.step(action)

        reward = step.reward
        self._state = self._convert_observation(step.observation)
        absorbing = step.last()

        return self._state, reward, absorbing, {}

    def render(self):
        img = self.env.physics.render(self._height_screen,
                                      self._width_screen,
                                      self._camera_id)
        self._viewer.display(img)

    def stop(self):
        self._viewer.close()

    @staticmethod
    def _convert_observation_space_vector(observation_space):
        observation_shape = 0
        for i in observation_space:
            shape = observation_space[i].shape
            observation_var = 1
            for dim in shape:
                observation_var *= dim
            observation_shape += observation_var

        return Box(low=-np.inf, high=np.inf, shape=(observation_shape,))


    @staticmethod
    def _convert_observation_space_pixels(observation_space):
        img_size = observation_space['pixels'].shape
        return Box(low=0., high=255.,
            shape=(3, img_size[0], img_size[1]))

    @staticmethod
    def _convert_action_space(action_space):
        low = action_space.minimum
        high = action_space.maximum

        return Box(low=np.array(low), high=np.array(high))

    @staticmethod
    def _convert_observation_vector(observation):
        obs = list()
        for i in observation:
            obs.append(np.atleast_1d(observation[i]).flatten())

        return np.concatenate(obs)

    @staticmethod
    def _convert_observation_pixels(observation):
        return observation['pixels'].transpose((2, 0, 1))
