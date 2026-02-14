from google.genai import types
from agent_lib import AbstractAgent


class GeminiAgent(AbstractAgent):

    def make_call(self, tool_call):
        name = tool_call.name
    
        if name in self.tool_index:
            func = self.tool_index[name]
            arguments = tool_call.args
            result = func(**arguments)
            if not isinstance(result, dict):
                result = {"result": result}
        else:
            result = {
                "error": f"Tool '{name}' not found in tools index"
            }
    
        return types.Part(
            function_response=types.FunctionResponse(
                name=name,
                response=result,
                id=tool_call.id,
            )
        )

    def loop(self, user_prompt, message_history=None):
        user_prompt_part = types.Content(
            parts=[types.Part(text=user_prompt)],
            role="user"
        )

        if message_history is None:
            message_history = [user_prompt_part]
        else:
            message_history.append(user_prompt_part)
        
        iteration_number = 1
        
        while True:
            print(f'iteration number {iteration_number}...')
        
            response = self.llm_client.models.generate_content(
                model=self.model,
                contents=message_history,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=self.tools)],
                    system_instruction=self.instructions,
                )
            )
        
            parts = response.candidates[0].content.parts
            message_history.append(types.Content(role="model", parts=parts))
            
            tool_call_results = []
        
            for part in parts:
                if part.function_call is not None:
                    function_call = part.function_call
                    print(f'executing {function_call.name}({function_call.args})...')
                    tool_call_output = self.make_call(function_call)
                    tool_call_results.append(tool_call_output)
        
                if part.text is not None:
                    text = part.text
                    print('ASSISTANT:', text)
        
            if len(tool_call_results) > 0:
                message_history.append(types.Content(role="user", parts=tool_call_results))
            else:
                break
        
            iteration_number = iteration_number + 1
            print()

        return message_history
