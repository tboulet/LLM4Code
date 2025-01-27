import os
import random
import re
import gymnasium as gym
import numpy as np
from openai import OpenAI
import tensorboardX
import tqdm


# Initialize environment and agent : the environment is a wagon
name_env = "maze"  # env coded in maze.py
code_env = open("src/maze.py").read()
tb_logger = tensorboardX.SummaryWriter(f"tensorboard/openai/{name_env}")

import src.maze as maze

env = maze.SimpleMaze(size=(10, 10), dynamic=False)


# Create the agent
class Agent:
    def __init__(self):
        # Initialize OpenAI API
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4o-mini"
        
        # Initialize the execution environment for the action function
        self.exec_globals = {}

        # Initialize prompt for asking for the action function
        self.formalism = (
            "Please reason step-by-step and analyze the given environment. "
            "After your reasoning, write a Python function action_function(observation, memory_dict) -> (action, memory_dict) that receives an observation as input and returns an action."
            "You are also free to use the memory_dict to store information between steps. This memory_dict is initialized as empty at the beginning of each episode and passed through each call to the action function."
            "The function should be returned in the following format:\n\n"
            "```python\n"
            "def action_function(observation, memory_dict):\n"
            "    # Your code here\n"
            "    return action, memory_dict\n"
            "```\n"
        )
        self.messages = []
        self.messages.append(
            {
                "role": "system",
                "content": f"You are an RL agent that needs to produce code that solves an RL environment. The code of the environment is the following:\n\n{code_env}.",
            }
        )
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Please write a Python function that solves the environment. "
                    f"{self.formalism}"
                ),
            }
        )

        # Ask the assistant for the action function
        self.ask_for_action_function()

    def reset(self):
        self.memory_dict = {}
        
    def act(self, observation):
        num_attempts = 5
        for i in range(num_attempts):
            try:
                # Execute the action function
                action, self.memory_dict = self.exec_globals["action_function"](observation, self.memory_dict)
                return action
            except Exception as e:
                # In case of an error, ask the assistant to redefine the function
                print(
                    f"Error in action function : {e}, trying to redefine it. ({i+1}/{num_attempts})"
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"An error happened when executing action_function. The observation was {observation}. The memory dict was {self.memory_dict}. The error was: {e}. "
                            "Please redefine the action function so that it works correctly."
                            f"{self.formalism}"
                        ),
                    }
                )
                self.ask_for_action_function()

        print("MESSAGES:")
        print(self.messages)
        raise ValueError("Action function could not be defined correctly.")

    def extract_function(self, response):
        """
        Extracts a Python function from a string containing reasoning and a code block.

        :param response: String, the assistant's response containing the reasoning and the function.
        :return: String, the extracted function definition or None if not found.
        """
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

    def ask_for_action_function(self):
        print(f"Asking the model {self.model} for action function...")
        # Ask the assistant for the action function
        answer_assistant = (
            self.client.chat.completions.create(
                model=self.model,  # Replace with your preferred model
                messages=self.messages,
            )
            .choices[0]
            .message.content
        )
        print(f"Answer of assistant: {answer_assistant}")

        # Extract the action function from the assistant's answer
        action_function = self.extract_function(answer_assistant)

        if action_function is not None:
            # Execute the code to define the action function
            exec(action_function, self.exec_globals)
            assert "action_function" in self.exec_globals and callable(
                self.exec_globals["action_function"]
            ), "Generated code does not define a callable 'action_function'."
        else:
            # Unsure an action function was defined earlier
            assert "action_function" in self.exec_globals, "No action function defined but this is required."
            
        # Add the answer generated to the messages
        self.messages.append(
            {
                "role": "assistant",
                "content": answer_assistant,
            }
        )


    def learn(self, cum_reward):
        # Inform the agent of the reward
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"The episode has ended with a cumulative reward of {cum_reward}."
                    "You can now try to propose an improved action function if you wish so. If so, please follow the same formalism as before."
                    "If you don't, you can simply not output any function in your answer."
                ),
            }
        )
        # Ask the assistant for the action function
        self.ask_for_action_function()


agent = Agent()


for ep in range(50):
    # Initialize environment
    print(f"Episode {ep}")
    observation, info = env.reset()
    agent.reset()
    terminated = False
    truncated = False
    cum_reward = 0
    t = 0
    tqdm_bar = tqdm.tqdm(range(100), desc="Running episode")

    # Run episode
    while not (terminated or truncated):
        env.render()
        action = agent.act(observation)
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error at {t} in step: {e}")
            raise

        # Logging
        tqdm_bar.update(1)
        t += 1
        cum_reward += reward

    # Learning
    agent.learn(cum_reward=cum_reward)

    tqdm_bar.close()
    tb_logger.add_scalar("total_reward", cum_reward, ep)
    print(f"Episode {ep} ended with a cumulative reward of {cum_reward}.")
