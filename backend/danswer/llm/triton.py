# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import sys
import numpy as np
import asyncio
import aiohttp
import json
from collections.abc import Iterator

from langchain.schema.language_model import LanguageModelInput

from danswer.configs.model_configs import GEN_AI_API_ENDPOINT
from danswer.configs.model_configs import GEN_AI_MAX_OUTPUT_TOKENS
from danswer.configs.model_configs import GEN_AI_MODEL_VERSION
from danswer.configs.model_configs import GEN_AI_TEMPERATURE
from danswer.configs.model_configs import GEN_AI_TRITON_PROTOCOL
from danswer.configs.chat_configs import QA_PROMPT_OVERRIDE  

from danswer.llm.interfaces import LLM
from danswer.llm.utils import convert_lm_input_to_basic_string
from danswer.utils.logger import setup_logger

import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *


logger = setup_logger()


class DanswerTriton(LLM):
    """
    This class provides support for connecting DAnswer to an Nivida Triton model inferencing server.
    The class leverages the gRPC protocol (default port :8001) to connect to Triton.
    """

    @property
    def requires_api_key(self) -> bool:
        return False

    def __init__(
        self,
        # Not used here but you probably want a model server that isn't completely open
        api_key: str | None,
        timeout: int,
        endpoint: str | None = GEN_AI_API_ENDPOINT,
        model_version: str | None = GEN_AI_MODEL_VERSION,
        triton_protocol: str | None = GEN_AI_TRITON_PROTOCOL,
        max_output_tokens: int = GEN_AI_MAX_OUTPUT_TOKENS,
        temperature: float = GEN_AI_TEMPERATURE,
        qa_prompt_override: str | None = QA_PROMPT_OVERRIDE
    ):
        if not endpoint:
            raise ValueError(
                "Cannot point Danswer to Triton LLM server without providing the "
                "endpoint for the model inferencing server."
            )

        self._endpoint = endpoint
        self._model_version = model_version
        self._triton_protocol = triton_protocol
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._qa_prompt_override = qa_prompt_override

    def _execute(self, input: LanguageModelInput) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if self._triton_protocol == 'grpc':
            prompts = [convert_lm_input_to_basic_string(input)]
            parameters = {
                "temperature": int(self._temperature) if self._temperature is not None else 0, #Can use temperature env parameter if interested.
                "max_tokens": self._max_output_tokens,
                "top_p": "1"
            }
            try:
                future = asyncio.ensure_future(self._triton_helper(prompts, sampling_params=parameters, streaming_mode=False))
                values = loop.run_until_complete(future)[0] #Only one response per request.
            except Exception as e:
                logger.info(f'DAnswer was unable to query the Triton server as a result of the following error: {e}')
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            response = ''
            if self._qa_prompt_override == 'weak':
                response = values.split("\n\nANSWER:")[1].strip()
            else:
                '''Direct & Chat QA each require a different formatted response. This is used to capture what prompt is asked and appropriately format the response.'''
                direct_qa_prompt_snippets = [
                    'Quote and cite relevant information from provided context based on the user query.',
                    'You are a text summarizing assistant that highlights the most important knowledge from the context provided',
                    'You ALWAYS responds with ONLY a JSON containing an answer and quotes that support the answer.'
                ]
                chat_init_prompt_snippets = [
                    'Given the conversation history and a follow up query, determine if the system should call an external search tool to better answer the latest user input.',
                    'Given the following conversation, provide a SHORT name for the conversation.',
                    'Given the following conversation and a follow up input, rephrase the follow up into a SHORT, standalone query (which captures any relevant context from previous messages) for a vectorstore.'
                ]
                chat_qa_prompt_snippets = [
                    'Do not provide any citations even if there are examples in the chat history.'
                ]

                if any(snippet in prompts[0] for snippet in direct_qa_prompt_snippets):
                    logger.info(f"PROMPT DIRECT QA: {prompts[0]}\nRESPONSE: {values}")
                    response = values.split("SAMPLE RESPONSE:")[1].strip()
                elif any(snippet in prompts[0] for snippet in chat_init_prompt_snippets):
                    logger.info(f"PROMPT CHAT INIT: {prompts[0]}")
                    if 'RESPONSE:' in values.strip():
                        response = values.split('RESPONSE:')[1]
                    else:
                        response = values.strip()
                elif any(snippet in prompts[0] for snippet in chat_qa_prompt_snippets):
                    logger.info(f"PROMPT CHAT QA: {prompts[0]}\nRESPONSE: {values}")
                    response = values.split("ANSWER:")[1].strip()
                elif 'You are a helper tool to determine if a query is answerable using retrieval augmented generation.' in prompts[0]:
                    response = values.split("THOUGHT:")[1].strip()
                else:
                    logger.info(f"PROMPT NOT CATCHED: {prompts[0]}\nRESPONSE:{values}")
                    response = [values.strip()]
            return response
        elif self._triton_protocol == 'rest':
            headers = {
                "Content-Type": "application/json",
            }

            '''The request body defined here is based on the config.pbtxt found in your respective Triton Server'''
            data = {
                "PROMPT": convert_lm_input_to_basic_string(input),
                "STREAM": False,
                "parameters": {
                    "temperature": int(self._temperature) if self._temperature is not None else 0,
                    "max_tokens": self._max_output_tokens,
                    "top_p": "1"
                }
            }
            try:
                full_endpoint = f'{self._endpoint}/v2/models/{self._model_version}/generate'
                future = asyncio.ensure_future(self._send_http_request(endpoint=full_endpoint, headers=headers, data=data))
                values = loop.run_until_complete(future)
            except Exception as e:
                logger.info(f'DAnswer was unable to query the Triton server as a result of the following error: {e}')
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()    
            
            response = ''
            if self._qa_prompt_override == 'weak':
                response = values.split("\n\nANSWER:")[1].strip() #Return only the answer, not the full prompt.
            else:
                '''Direct & Chat QA each require a different formatted response. This is used to capture what prompt is asked and appropriately format the response.'''
                direct_qa_prompt_snippets = [
                        'Quote and cite relevant information from provided context based on the user query.',
                        'You are a text summarizing assistant that highlights the most important knowledge from the context provided',
                        'You are a question answering system that is constantly learning and improving.'
                ]
                chat_qa_prompt_snippets = [
                        'Given the conversation history and a follow up query, determine if the system should call an external search tool to better answer the latest user input.',
                        'Given the following conversation, provide a SHORT name for the conversation.'
                ]
                if any(snippet in prompts[0] for snippet in direct_qa_prompt_snippets):
                    response = values.split("SAMPLE RESPONSE:")[1].strip()
                elif any(snippet in prompts[0] for snippet in chat_qa_prompt_snippets):
                    response = values.strip()
                elif 'You are a helper tool to determine if a query is answerable using retrieval augmented generation.' in prompts[0]:
                    response = values.split("THOUGHT:")[1].strip()
                else:
                    response = [values.strip()]
            return response

    def log_model_configs(self) -> None:
        logger.debug(f"Custom model at: {self._endpoint}")

    def invoke(self, prompt: LanguageModelInput) -> str:
        return self._execute(prompt)

    def stream(self, prompt: LanguageModelInput) -> Iterator[str]:
        return self._execute(prompt)
        
    def _send_grpc_request(self, prompt, stream, request_id, sampling_parameters, model_name, send_parameters_as_tensor=True):
        '''
        Description:
            A function that handles the forwarding of an inference request to the Triton server via gRPC protocol.
            This function is called by the triton_helper function.
        Parameters:
            :param: prompt: string: The prompt to be evaluated by the model (i.e., model input).
            :param: stream: bool: Streaming response or not? The default is 'False' in triton_helper function.
            :param: request_id: str: An identifier for the inference request.
            :param: sampling_parameters: dict: A dictionary of hyperparameters to pass to the model at inference time. Refer to example below:
                - Example: {"temperature": "0", "top_p": "1", "frequency_penalty":"1.15", "max_tokens":"4096"}
            :param: model_name: str: Name of model in triton server. The default is 'vllm' in triton_helper function (don't change this unless you know what you're doing). 
            :param: send_parameters_as_tensor: int [Optional]: Whether the parameters should be sent as a tensor or not. The default is 'True'.
        Return:
            :return: dict: A dictionary containing the model name, inference inputs, inference outputs, request id, and sampling parameters.
        '''
        inputs = []
        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        try:
            inputs.append(grpcclient.InferInput("PROMPT", [1], "BYTES")) #PROMPT may need to be changed based on your config.pbtxt file in your Triton Server
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as e:
            print(f"Encountered an error {e}")

        stream_data = np.array([stream], dtype=bool)
        inputs.append(grpcclient.InferInput("STREAM", [1], "BOOL")) #STREAM may need to be changed based on your config.pbtxt file in your Triton Server
        inputs[-1].set_data_from_numpy(stream_data)

        # Request parameters are not yet supported via BLS. Provide an
        # optional mechanism to send serialized parameters as an input
        # tensor until support is added

        if send_parameters_as_tensor:
            sampling_parameters_data = np.array(
                [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
            )
            inputs.append(grpcclient.InferInput("SAMPLING_PARAMETERS", [1], "BYTES")) #SAMPLING_PARAMETERS may need to be changed based on your config.pbtxt file in your Triton Server
            inputs[-1].set_data_from_numpy(sampling_parameters_data)

        # Add requested outputs
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("TEXT")) #TEXT may need to be changed based on your config.pbtxt file in your Triton Server

        # Issue the asynchronous sequence inference.
        return {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(request_id),
            "parameters": sampling_parameters,
        }

    async def _send_http_request(self, endpoint, headers, data):
        '''
        Description:
            A function that handles the forwarding of an inference request to the Triton server via rest protocol.
        Parameters:
            :param: endpoint: string: The HTTP endpoint for the Triton Server (e.g. http://my-triton-server-endpoint:8000/v2/models/{model_name}/generate).
            :param: headers: dict: The HTTP request headers for the request.
            :param: data: str: The request body for the HTTP request.
        Return:
            :return: string: The model output to the inference request.
        '''        
        async with aiohttp.ClientSession() as session:
            async with session.post(url=endpoint, headers=headers, data=json.dumps(data)) as resp:
                response = await resp.json()
                return response['TEXT'] # Dependent on your config.pbtxt file in your Triton server.

    async def _triton_helper(self, prompts, sampling_params, streaming_mode=True, verbose=False):
        '''
        Description:
            The primary Triton orchestration function that handles the instantiation of the GRPC client and the caller of the 'send_request' function.
            This function is asynchronous, designed to have multiple instantiations that wait to hear a response back from the Triton server.
        Parameters:
            :param: prompt: string: The prompt to be evaluated by the model (i.e., model input).
            :param: sampling_parameters: dict: A dictionary of hyperparameters to pass to the model at inference time. Refer to example below:
                - Example: {"temperature": "0", "top_p": "1", "frequency_penalty":"1.15", "max_tokens":"4096"}
            :param: streaming_mode: bool: Streaming response or not? The default is 'False'.
            :param: verbose: bool: Do you want a verbose output or not? The default is 'False'.
        Return:
            :return: response_list: list of dicts: A list of dictionaries from the send_request route. These dictionaries contain the model name, inference inputs, inference outputs, request id, and sampling parameters.
        '''
        model_name = self._model_version
        host = self._endpoint
        results_dict = {}
        async with grpcclient.InferenceServerClient(url=host, verbose=verbose) as triton_client:
            # Request iterator that yields the next request
            iterations= 1 # Number of iterations through prompts. Default 1.
            offset = 0 # Add offset to request IDs used. Default 0.
            async def async_request_iterator():
                try:
                    for iter in range(iterations):
                        for i, prompt in enumerate(prompts):
                            prompt_id = offset + (len(prompts) * iter) + i
                            results_dict[str(prompt_id)] = []
                            yield self._send_grpc_request(
                                prompt, streaming_mode, prompt_id, sampling_params, model_name
                            )
                except Exception as error:
                    print(f"caught error in request iterator:  {error}")

            try:
                # Start streaming
                response_iterator = triton_client.stream_infer(
                    inputs_iterator=async_request_iterator(),
                    stream_timeout= self._timeout #Stream timeout in seconds. Default is None.,
                )
                # Read response from the stream
                async for response in response_iterator:
                    result, error = response
                    if error:
                        print(f"Encountered error while processing: {error}")
                    else:
                        output = result.as_numpy("TEXT") #TEXT may need to be changed based on your config.pbtxt file in your Triton Server
                        for i in output:
                            results_dict[result.get_response().id].append(i)
            except InferenceServerException as error:
                print(error)
                sys.exit(1)

        response_list = []
        for id in results_dict.keys():
            for result in results_dict[id]:
                decoded_result = result.decode("utf-8")
                response_list.append(decoded_result)

        if verbose:
            print(f"\nRESULTS: `{response_list}` ===>")
            
        return response_list    

