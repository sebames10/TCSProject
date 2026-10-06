'''MCP server for AI Lab training operations.

Exposes tools, one resource, and one prompt over the stdio transport.

Run it directly for a syntax check:  python training_server.py
(it will block, waiting for a host to speak first — that is correct)
'''

from mcp.server import MCPServer

mcp = MCPServer(
    'Training Ops',
    instructions='Training operations for the AI Lab. Use the math tools for any calculation.',
)


# ---------------------------------------------------------------------------
# Tools — the model can request these
# ---------------------------------------------------------------------------

@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    '''Add two numbers and return the result.'''
    return a + b


@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    '''Multiply two numbers and return the result.'''
    return a * b


@mcp.tool()
def divide_numbers(a: float, b: float) -> dict:
    '''Divide a by b and return the result.'''
    if b == 0:
        # Same idea as Lesson 4: a tool that fails returns a structured error
        # the model can read and recover from, instead of crashing the call.
        return {'error': 'Cannot divide by zero. Ask the user for a non-zero divisor.'}
    return {'result': a / b}


ROOMS = {
    'lab_a': {'capacity': 12, 'has_projector': True},
    'lab_b': {'capacity': 24, 'has_projector': True},
    'lab_c': {'capacity': 6, 'has_projector': False},
}


@mcp.tool()
def get_room_capacity(room_id: str) -> dict:
    '''Gets info.'''  ### FIXME: would a model pick this tool with a description like this?
    room = ROOMS.get(room_id)

    if room is None:
        return {'found': False, 'room_id': room_id, 'error': 'Room not found.'}

    return {
        'found': True,
        'room_id': room_id,
        'capacity': room['capacity'],
        'has_projector': room['has_projector'],
    }


# ---------------------------------------------------------------------------
# Resource — the application can read this (no model involved)
# ---------------------------------------------------------------------------

@mcp.resource('training://catalog')
def training_catalog() -> str:
    '''The AI Lab course catalogue.'''
    return (
        '1. Web and FastAPI\n'
        '2. LLM Inference Parameters\n'
        '3. RAG from Scratch\n'
        '4. Function Calling and Tool Usage\n'
        '5. LangChain Tool Calling\n'
        '6. LangGraph Calculator\n'
        '7. LangGraph RAG Agent\n'
    )


# ---------------------------------------------------------------------------
# Prompt — the user can pick this
# ---------------------------------------------------------------------------

@mcp.prompt()
def training_plan(topic: str) -> str:
    '''Draft a short training plan for one topic.'''
    return (
        f'Draft a 90-minute training plan for the topic: {topic}. '
        'Include learning objectives, one hands-on exercise, and one discussion question.'
    )


if __name__ == '__main__':
    # No arguments means stdio: the host launches this file as a subprocess.
    mcp.run()
