'''Minimal MCP client: connect to training_server.py, list, and call.'''

import asyncio
import sys

from mcp import Client, StdioServerParameters
from mcp.types import TextContent, TextResourceContents

SERVER = StdioServerParameters(
    command=sys.executable,
    args=['training_server.py'],
)


async def main() -> None:
    # Entering the block spawns the subprocess and negotiates; leaving it shuts it down.
    async with Client(SERVER) as client:
        name = client.server_info.name if client.server_info else 'unknown'
        print('Connected to:', name)
        print('Protocol version:', client.protocol_version)
        print('Instructions:', client.instructions)
        print()

        # ---- Tools: what the model would see -------------------------------
        listed = await client.list_tools()
        print(f'Tools available ({len(listed.tools)}):')
        for tool in listed.tools:
            print(f'  - {tool.name}: {tool.description}')
        print()

        # ---- Resources: what the application can read ----------------------
        resources = await client.list_resources()
        print('Resources:', [resource.uri for resource in resources.resources])

        read = await client.read_resource('training://catalog')
        print('Catalog read from the server:')
        for contents in read.contents:
            if isinstance(contents, TextResourceContents):
                print(contents.text)
        print()

        # ---- Prompts: what the user can pick -------------------------------
        prompts = await client.list_prompts()
        print('Prompts:', [prompt.name for prompt in prompts.prompts])

        rendered = await client.get_prompt('training_plan', {'topic': 'MCP'})
        for message in rendered.messages:
            if isinstance(message.content, TextContent):
                print(f'  [{message.role}] {message.content.text}')
        print()

        # ---- Calling a tool -------------------------------------------------
        ok = await client.call_tool('get_room_capacity', {'room_id': 'lab_b'})
        print('call_tool(get_room_capacity, lab_b)')
        print('  is_error          :', ok.is_error)
        print('  structured_content:', ok.structured_content)
        print()

        missing = await client.call_tool('get_room_capacity', {'room_id': 'lab_z'})
        print('call_tool(get_room_capacity, lab_z)')
        print('  is_error          :', missing.is_error)
        print('  structured_content:', missing.structured_content)
        print()

        unknown = await client.call_tool('not_a_tool', {})
        print('call_tool(not_a_tool)')
        print('  is_error:', unknown.is_error)
        print('  content :', unknown.content[0].text)


if __name__ == '__main__':
    asyncio.run(main())
