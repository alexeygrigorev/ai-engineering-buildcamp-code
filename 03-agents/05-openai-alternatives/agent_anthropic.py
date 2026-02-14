import json
from agent_lib import AbstractAgent

class AnthropicAgent(AbstractAgent):

    def make_call(self, tool_call):
        name = tool_call.name
    
        if name in self.tool_index:
            func = self.tool_index[name]
            arguments = tool_call.input
            result_raw = func(**arguments)
            result = json.dumps(result_raw)
        else:
            result = f"Tool '{name}' not found in tools index"
    
        return self.format_tool_call_output(tool_call.id, result)
    
    def format_tool_call_output(self, tool_call_id, tool_call_output):
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": tool_call_output
        }

    def loop(self, user_prompt, message_history=None):
        if message_history is None:
            message_history = [{"role": "user", "content": user_prompt}]
        else:
            message_history.append({"role": "user", "content": user_prompt})

        iteration_number = 1
        
        while True:
        
            response = self.llm_client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.instructions,
                messages=message_history,
                tools=self.tools,
            )
            
            print(f'iteration number {iteration_number}...')
            message_history.append({"role": "assistant", "content": response.content})
        
            tool_call_results = []
        
            for message in response.content:
                if message.type == 'tool_use':
                    print(f'executing {message.name}({message.input})...')
                    tool_call_output = self.make_call(message)
                    tool_call_results.append(tool_call_output)
            
                if message.type == 'text':
                    text = message.text
                    print('ASSISTANT:', text)
        
        
            if len(tool_call_results) > 0:
                message_history.append({"role": "user", "content": tool_call_results})
            else:
                break
        
            iteration_number = iteration_number + 1
            print()
        
        return message_history
