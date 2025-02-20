import traceback
import io
import sys

code = """
def raise_error():
    0/0
    raise Exception("Error")

def print_hello():
    print("Hello")
    raise_error()
"""

exec_globals = {}
exec(code, exec_globals)

func = exec_globals["print_hello"]

try:
    func()
except Exception as e:
    
    print(f"Error: {e}\n")
    
    # Capture the stack trace, focusing on the actual exception
    stack_trace = traceback.format_exc()
    print(f"Stack Trace:\n{stack_trace}\n")
    
    # Clean up the stack trace to remove the exec context frames
    relevant_stack_trace = "\n".join(line for line in stack_trace.splitlines() if "<string>" in line)
    print(f"Relevant Stack Trace:\n{relevant_stack_trace}\n")
    
    # Combine the output and error information
    full_error_info = f"Stack Trace:\n{relevant_stack_trace} \nError Output:\n{e}"
    print("\n" + full_error_info)
