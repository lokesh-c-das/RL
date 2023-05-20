from environment import Environment
from agent import Agent

def main():
    # initialize
    env = Environment("Q-Learning", 'human')
    agent = Agent(env)
    agent.train()


# call main()
if __name__ == '__main__':
    main()
