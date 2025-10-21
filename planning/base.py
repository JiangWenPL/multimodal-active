class PolicyBase:
    def __init__(self) -> None:
        self.goal = None

    def init(self, test_ds, episode_id):
        """ Init the policy from episode """
        pass

    def act(self, **obs):
        """
        Generate the next action based on the observation and stage goal
        """
        pass

    def save(self, path):
        """ Save the policy to the given path """
        pass

    def load(self, path):
        """ Load the policy from the given path """
        pass

    def set_next_goal(self, goal):
        """ 
        Set the next goal for the agent
        """
        pass