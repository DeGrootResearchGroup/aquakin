`aquakin mcp --allow-python` lets an AI assistant run Python scripts it writes
in your project folder (`run_python`), so apps that cannot run code themselves,
such as Claude Desktop, can use any part of aquakin. The tool is off unless
the flag is given, the assistant's app asks before each run, a script is
stopped after a timeout, and it can be kept in the project to run again.
