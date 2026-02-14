import json
from agent_lib import AbstractAgent


class OpenAIChatCompletionsAgent(AbstractAgent):

    def __init__(self, llm_client, model, instructions, tool_pairs):
        self.llm_client = llm_client
        self.model = model
        self.instructions = instructions
        self.tool_index = {t["function"]["name"]: f for (f, t) in tool_pairs}
        self.tools = [t for (_, t) in tool_pairs]

    def make_call(self, tool_call):
        function = tool_call.function
        name = function.name

        if name in self.tool_index:
            func = self.tool_index[name]
            arguments = json.loads(function.arguments)
            result = func(**arguments)
            result_json = json.dumps(result)
        else:
            result_json = f"Tool '{name}' not found in tools index"

        return self.format_tool_call_output(
            tool_call_id=tool_call.id,
            tool_call_output=result_json,
        )

    def format_tool_call_output(self, tool_call_id, tool_call_output):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_call_output,
        }

    def loop(self, user_prompt, message_history=None):
        if not message_history:
            message_history = [
                {"role": "system", "content": self.instructions},
            ]

        message_history.append({"role": "user", "content": user_prompt})
        
        iteration_number = 1
        
        while True:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=message_history,
                tools=self.tools,
            )
        
            print(f'iteration number {iteration_number}...')
            message = response.choices[0].message
            message_history.append(message)
        
            if message.content:
                print('ASSISTANT:', message.content)
            
            has_function_calls = (message.tool_calls is not None) and \
                (len(message.tool_calls) > 0)
            
            if has_function_calls:
                for tool_call in message.tool_calls:
                    function = tool_call.function
                    print(f'executing {function.name}({function.arguments})...')
                    tool_call_output = self.make_call(tool_call)
                    message_history.append(tool_call_output)
        
            if not has_function_calls:
                break
        
            iteration_number = iteration_number + 1
            print()

        return message_history