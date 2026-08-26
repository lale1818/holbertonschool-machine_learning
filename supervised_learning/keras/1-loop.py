#!/usr/bin/env python3
"""
Module that prompts the user in a continuous loop for questions.
"""


def question_loop():
    """
    Prompts the user with 'Q: ' and prints 'A: ' as a response.
    Exits gracefully when the user inputs 'exit', 'quit', 'goodbye', or 'bye'.
    """
    exit_commands = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        try:
            user_input = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("A: Goodbye")
            break

        if user_input.lower() in exit_commands:
            print("A: Goodbye")
            break
        else:
            print("A:")


if __name__ == "__main__":
    question_loop()
