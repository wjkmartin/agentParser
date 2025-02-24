import json
import logging
import re
from collections import defaultdict

RATIONALE_VALUE_REGEX_LIST = [
    "<thinking>(.*?)(</thinking>)",
    "(.*?)(</thinking>)",
    "(<thinking>)(.*?)"
]
RATIONALE_VALUE_PATTERNS = [re.compile(
    regex, re.DOTALL) for regex in RATIONALE_VALUE_REGEX_LIST]

FINAL_RESPONSE_REGEX = r"<final_response>([\s\S]*?)<final_response>"
FINAL_RESPONSE_PATTERN = re.compile(FINAL_RESPONSE_REGEX, re.DOTALL)

ANSWER_REGEX = r"(?<=<answer>)(.*)"
ANSWER_PATTERN = re.compile(ANSWER_REGEX, re.DOTALL)

ANSWER_TAG = "<answer>"
ASK_USER = "user__askuser"

KNOWLEDGE_STORE_SEARCH_ACTION_PREFIX = "x_amz_knowledgebase_"

ANSWER_PART_REGEX = "<answer_part>(.+?)</answer_part>"
ANSWER_TEXT_PART_REGEX = "<text>(.+?)</text>"
ANSWER_REFERENCE_PART_REGEX = "<source>(.+?)</source>"

ANSWER_PART_PATTERN = re.compile(ANSWER_PART_REGEX, re.DOTALL)
ANSWER_TEXT_PART_PATTERN = re.compile(ANSWER_TEXT_PART_REGEX, re.DOTALL)
ANSWER_REFERENCE_PART_PATTERN = re.compile(
    ANSWER_REFERENCE_PART_REGEX, re.DOTALL)

MISSING_API_INPUT_FOR_USER_REPROMPT_MESSAGE = (
    "Missing the parameter 'question' for user__askuser function call. Please try again with the correct argument added."
)
FUNCTION_CALL_STRUCTURE_REPROMPT_MESSAGE = (
    "The tool name format is incorrect. The format for the tool name must be: 'httpVerb__actionGroupName__apiName."
)
logger = logging.getLogger()


def lambda_handler(event, context):
    logger.setLevel("INFO")
    logger.info(f"Lambda input event: {json.dumps(event)}")
    raw_response = event.get('invokeModelRawResponse', '')

    if event.get("promptType") == "KNOWLEDGE_BASE_RESPONSE_GENERATION":
        return handle_kb_response(raw_response)

    try:
        response = load_response(raw_response)
        if isinstance(response, dict) and "stop_reason" in response:
            stop_reason = response["stop_reason"]
            content = response["content"]
            content_by_type = get_content_by_type(content)
            rationale = parse_rationale(content_by_type)
            parsed_response = {
                'promptType': "ORCHESTRATION",
                'orchestrationParsedResponse': {
                    'rationale': rationale
                }
            }

            logger.info(f"Stop reason: {stop_reason}")

            match stop_reason:
                case 'tool_use':
                    # Check if there is an ask user
                    try:
                        ask_user = parse_ask_user(content_by_type)
                        if ask_user:
                            parsed_response['orchestrationParsedResponse']['responseDetails'] = {
                                'invocationType': 'ASK_USER',
                                'agentAskUser': {
                                    'responseText': ask_user,
                                    'id': content_by_type['tool_use'][0]['id']
                                },
                            }
                            logger.info(
                                f"Lambda output (ask user): {json.dumps(parsed_response)}")
                            return parsed_response
                    except ValueError as e:
                        addRepromptResponse(parsed_response, e)
                        return parsed_response
                    try:
                        parsed_response = parse_function_call(
                            content_by_type, parsed_response)
                        logger.info(
                            f"Lambda output (function call): {json.dumps(parsed_response)}")
                        return parsed_response
                    except ValueError as e:
                        addRepromptResponse(parsed_response, e)
                        return parsed_response

                case 'end_turn' | 'stop_sequence':
                    try:
                        if content_by_type["text"]:
                            for text_content in content_by_type["text"]:
                                final_answer, generated_response_parts = parse_answer(
                                    text_content)
                                if final_answer:
                                    parsed_response['orchestrationParsedResponse'][
                                        'responseDetails'] = {
                                        'invocationType': 'FINISH',
                                        'agentFinalResponse': {
                                            'responseText': final_answer
                                        }
                                    }

                                    if generated_response_parts:
                                        parsed_response['orchestrationParsedResponse']['responseDetails'][
                                            'agentFinalResponse']['citations'] = {
                                            'generatedResponseParts': generated_response_parts
                                        }

                                    logger.info(
                                        "Final answer parsed response: " + str(parsed_response))
                                return parsed_response
                    except ValueError as e:
                        addRepromptResponse(parsed_response, e)
                        return parsed_response
                case _:
                    addRepromptResponse(
                        parsed_response, 'Failed to parse the LLM output')
                    logger.info(
                        f"Lambda output (error): {json.dumps(parsed_response)}")
                    return parsed_response
        else:
            return handle_kb_response(raw_response)
    except Exception:
        return handle_kb_response(raw_response)


def handle_kb_response(raw_response):
    parsed_response = {
        'promptType': 'KNOWLEDGE_BASE_RESPONSE_GENERATION',
        'knowledgeBaseResponseGenerationParsedResponse': {
            'generatedResponse': parse_kb_generated_response(raw_response)
        }
    }
    logger.info(f"Lambda output (kb response): {json.dumps(parsed_response)}")
    return parsed_response


def load_response(text):
    raw_text = r'{}'.format(text)
    json_text = json.loads(raw_text)
    return json_text


def get_content_by_type(content):
    content_by_type = defaultdict(list)
    for content_value in content:
        content_by_type[content_value["type"]].append(content_value)
    return content_by_type


def parse_rationale(content_by_type):
    if "text" in content_by_type:
        rationale = content_by_type["text"][0]["text"]
        if rationale is not None:
            rationale_matcher = next(
                (pattern.search(rationale)
                 for pattern in RATIONALE_VALUE_PATTERNS if pattern.search(rationale)),
                None
            )
            if rationale_matcher:
                rationale = rationale_matcher.group(1).strip()
        return rationale
    return None


def parse_answer(response):
    text = response["text"].strip()
    logger.info(f"Parsing answer. Input: {text}")

    if has_generated_response(text):
        logger.info("Generated response format detected")
        return parse_generated_response(text)
    answer_match = ANSWER_PATTERN.search(text)
    if answer_match:
        answer = answer_match.group(0).strip()
        logger.info(f"Found direct answer match: {answer}")
        return answer, None
    logger.info("No answer pattern found")
    return None, None


def parse_generated_response(text):
    logger.info("Parsing generated response")
    results = []
    answer_parts = list(ANSWER_PART_PATTERN.finditer(text))
    logger.info(f"Found {len(answer_parts)} answer parts")
    for match in answer_parts:
        part = match.group(1).strip()
        text_match = ANSWER_TEXT_PART_PATTERN.search(part)
        if not text_match:
            logger.error("Failed to find text match in answer part")
            raise ValueError("Could not parse generated response")
        text_content = text_match.group(1).strip()
        logger.info(f"Found text content: {text_content[:100]}...")
        references = parse_references(text, part)
        results.append((text_content, references))
    final_response = " ".join([r[0] for r in results])
    logger.info(f"Generated final response length: {len(final_response)}")
    generated_response_parts = []
    for text_content, references in results:
        generatedResponsePart = {
            'text': text_content,
            'references': references
        }
        generated_response_parts.append(generatedResponsePart)
    return final_response, generated_response_parts


def has_generated_response(raw_response):
    has_parts = ANSWER_PART_PATTERN.search(raw_response) is not None
    logger.info(f"Generated response format detected: {has_parts}")
    return has_parts


def parse_references(raw_response, answer_part):
    logger.info(f"Parsing references from answer part: {answer_part}")
    references = []
    for match in ANSWER_REFERENCE_PART_PATTERN.finditer(answer_part):
        reference = match.group(1).strip()
        logger.info(f"Found reference: {reference}")
        references.append({'sourceId': reference})
    logger.info(f"Total references found: {len(references)}")
    return references


def parse_ask_user(content_by_type):
    try:
        if content_by_type["tool_use"][0]["name"] == ASK_USER:
            ask_user_question = content_by_type["tool_use"][0]["input"]["question"]
            if not ask_user_question:
                raise ValueError(MISSING_API_INPUT_FOR_USER_REPROMPT_MESSAGE)
            return ask_user_question
    except ValueError as ex:
        raise ex
    return None


def parse_function_call(content_by_type, parsed_response):
    logger.info(
        f"Parsing function call. Input content: {json.dumps(content_by_type)}")
    try:
        content = content_by_type["tool_use"][0]
        tool_name = content["name"]

        action_split = tool_name.split('__')
        verb = action_split[0].strip()
        resource_name = action_split[1].strip()
        function = action_split[2].strip()
    except ValueError as ex:
        raise ValueError(FUNCTION_CALL_STRUCTURE_REPROMPT_MESSAGE)

    parameters = {}
    for param, value in content["input"].items():
        parameters[param] = {'value': value}

    parsed_response['orchestrationParsedResponse']['responseDetails'] = {}

    # Function calls can either invoke an action group or a knowledge base.
    # Mapping to the correct variable names accordingly
    if resource_name.lower().startswith(KNOWLEDGE_STORE_SEARCH_ACTION_PREFIX):
        parsed_response['orchestrationParsedResponse']['responseDetails'][
            'invocationType'
        ] = 'KNOWLEDGE_BASE'
        parsed_response['orchestrationParsedResponse']['responseDetails'][
            'agentKnowledgeBase'
        ] = {
            'searchQuery': parameters['searchQuery'],
            'knowledgeBaseId': resource_name.replace(KNOWLEDGE_STORE_SEARCH_ACTION_PREFIX, ''),
            'id': content["id"]
        }
        logger.info(
            f"Knowledge base tool use response: {json.dumps(parsed_response)}")
        return parsed_response

    if isinstance(value, list):
        parameters[param] = {'value': ','.join(str(v) for v in value)}

    parsed_response['orchestrationParsedResponse']['responseDetails'][
        'invocationType'
    ] = 'ACTION_GROUP'
    parsed_response['orchestrationParsedResponse']['responseDetails'][
        'actionGroupInvocation'
    ] = {
        "verb": verb,
        "actionGroupName": resource_name,
        "apiName": function,
        "actionGroupInput": parameters,
        "id": content["id"]
    }
    logger.info(
        f"Action group tool use response: {json.dumps(parsed_response)}")
    return parsed_response


def addRepromptResponse(parsed_response, error):
    error_message = str(error)
    logger.warn(error_message)
    parsed_response['orchestrationParsedResponse']['parsingErrorDetails'] = {
        'repromptResponse': error_message
    }


def parse_kb_generated_response(sanitized_llm_response):
    logger.info(
        f"Parsing KB generated response. Input: {json.dumps(sanitized_llm_response)}")
    results = []
    for match in ANSWER_PART_PATTERN.finditer(sanitized_llm_response):
        part = match.group(1).strip()
        text_match = ANSWER_TEXT_PART_PATTERN.search(part)
        if not text_match:
            logger.error("Could not find text match in answer part")
            raise ValueError("Could not parse generated response")
        text = text_match.group(1).strip()
        references = parse_references(sanitized_llm_response, part)
        results.append((text, references, part))
    generated_response_parts = []
    for text, references, part in results:
        full_text = f"{text} {part}"
        generatedResponsePart = {
            'text': full_text,
            'references': references
        }
        generated_response_parts.append(generatedResponsePart)
    response = {
        'generatedResponseParts': generated_response_parts
    }
    logger.info(f"KB generated response output: {json.dumps(response)}")
    return response
