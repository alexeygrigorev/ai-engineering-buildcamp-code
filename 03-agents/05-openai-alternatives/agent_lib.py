import json


class AbstractAgent:

    def __init__(self, llm_client, model, instructions, tool_pairs):
        self.llm_client = llm_client
        self.model = model
        self.instructions = instructions
        self.tool_index = {d['name']: f for (f, d) in tool_pairs}
        self.tools = [d for (_, d) in tool_pairs]

    def make_call(self, tool_call):
        arguments = json.loads(tool_call.arguments)
        name = tool_call.name

        if name in self.tool_index:
            func = self.tool_index[name]
            result_raw = func(**arguments)
            result_json = json.dumps(result_raw)
        else:
            result_json = f'function {name} does not exist'

        return self.format_tool_call_output(
            tool_call_id=tool_call.call_id,
            tool_call_output=result_json
        )

    def format_tool_call_output(self, tool_call_id, tool_call_output):
        raise Exception('not implemented')

    def loop(self, user_prompt, message_history=None):
        raise Exception('not implemented')     

    def qna(self):
        message_history = []

        while True:
            user_prompt = input('You:')
            if user_prompt.lower().strip() == 'stop':
                break
            
            message_history = self.loop(user_prompt, message_history)

        return message_history