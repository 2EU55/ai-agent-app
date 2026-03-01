import sys
import os
sys.path.append(os.path.join(os.getcwd(), "pylib"))
print("Importing langchain_classic...")
try:
    import langchain_classic
    print("Success: import langchain_classic")
except Exception as e:
    print(f"Error: {e}")

print("Importing langchain_classic.agents...")
try:
    import langchain_classic.agents
    print("Success: import langchain_classic.agents")
except Exception as e:
    print(f"Error: {e}")
