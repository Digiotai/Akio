# digiotai_jazz/agent.py
class Agent:
    def __init__(self, expertise, task, input_type, output_type):
        self.expertise = expertise
        self.task = task
        self.input_type = input_type
        self.output_type = output_type

    def __str__(self):
        return f"You are {self.expertise}, your tasks it to {self.task} I will give you {self.input_type} give me a {self.output_type} ####"
