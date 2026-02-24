import os

import dotenv
dotenv.load_dotenv()

from logfire.query_client import LogfireQueryClient

from models import RAGResponse
from trace_replay import fetch_trace, fetch_traces, trace_to_run_result

client = LogfireQueryClient(read_token=os.environ['LOGFIRE_READ_TOKEN'])

# --- Single trace ---
trace_id = '019c8f29a1ad902bdd2df975fb78d117'

trace = fetch_trace(trace_id, client)
run = trace_to_run_result(trace, output_type=RAGResponse)

assert isinstance(run.output, RAGResponse)
print(f'Single trace: {type(run.output).__name__}, confidence={run.output.confidence}, {len(run.all_messages())} messages')

# --- Batch fetch ---
trace_ids = [
    '019c8f3f6447f880e1a12852e646af68',
    '019c8f3d662b1302eeb88ce03a1504a2',
    '019c8f3a23e4cc6bd259a5a00f708b29',
    '019c8f2d3ef001f77db918166936d315',
    '019c8f29a1ad902bdd2df975fb78d117',
]

traces = fetch_traces(trace_ids, client)
print(f'\nBatch fetch: {len(traces)}/{len(trace_ids)} traces found')

for tid, trace_data in traces.items():
    run = trace_to_run_result(trace_data, output_type=RAGResponse)
    print(f'  {tid[:12]}... {len(run.all_messages()):>3} messages, '
          f'{trace_data.input_tokens:>6} in / {trace_data.output_tokens:>5} out tokens, '
          f'confidence={run.output.confidence}')

print('\nOK')
