from environment import Environment
from agent import Agent

# Remove Warning
import warnings
warnings.filterwarnings('ignore')

def main():
    # initialize
    env = Environment("Q-Learning", 'human')
    agent = Agent(env)
    agent.train()


# call main()
if __name__ == '__main__':
    main()
