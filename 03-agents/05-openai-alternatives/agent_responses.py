from agent_lib import AbstractAgent


class OpenAIResponsesAgent(AbstractAgent):

    def format_tool_call_output(self, tool_call_id, tool_call_output):
        return {
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": tool_call_output,
        }

    def loop(self, user_prompt, message_history=None):
        if not message_history:
            message_history = [
                {"role": "system", "content": self.instructions},
            ]

        message_history.append({"role": "user", "content": user_prompt})

        iteration_number = 1

        while True:
            response = self.llm_client.responses.create(
                model=self.model,
                input=message_history,
                tools=self.tools,
            )

            print(f'iteration number {iteration_number}...')
            message_history.extend(response.output)

            has_function_calls = False

            for message in response.output:
                if message.type == 'function_call':
                    print(f'executing {message.name}({message.arguments})...')
                    tool_call_output = self.make_call(message)
                    message_history.append(tool_call_output)
                    has_function_calls = True

                if message.type == 'message':
                    text = message.content[0].text
                    print('ASSISTANT:', text)

            iteration_number = iteration_number + 1
            print()

            if not has_function_calls:
                break

        return message_history