import io
import json
import os
import time
from typing import Any

import requests
from openai import OpenAI
from openai.types.beta.assistant_stream_event import (ThreadMessageCompleted,
                                                      ThreadRunRequiresAction,
                                                      ThreadRunStepCompleted)
from openai.types.beta.threads.runs import (FileSearchToolCall,
                                            ToolCallsStepDetails)


class CodelldBot:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.openai = OpenAI()
        self.current_repository = os.getenv('GITHUB_REPOSITORY')
        self.search_repository = os.getenv('SEARCH_REPOSITORY') or self.current_repository
        self.modify = bool(self.token and os.getenv('MODIFY'))
        self.verbose = bool(os.getenv('VERBOSE_LOGGING'))
        self.assistant = self.openai.beta.assistants.retrieve(os.getenv('ASSISTANT_ID'))
        self.system_prompt, *self.step_prompts = self.assistant.instructions.split('\n---\n')
        self.found_issues = {}

    def handle_event(self):

        with open(os.getenv('GITHUB_EVENT_PATH'), 'rb') as f:
            event = json.load(f)

        if self.verbose:
            print('Event:', event)

        match os.getenv('GITHUB_EVENT_NAME'):
            case 'issues':
                issue = event['issue']
            case 'workflow_dispatch':
                issue_number = int(event['inputs']['issue_number'])
                response = self.github_request('GET', f'/repos/{self.current_repository}/issues/{issue_number}')
                if not response.ok:
                    raise Exception(f'''Could not fetch issue: {response['message']}''')
                issue = response.json()

        if self.verbose:
            print('Issue:', issue)

        issue_content = self.make_issue_content(issue, show_labels=False)
        issue_file = self.openai.files.create(
            file=('NEW_ISSUE.md', issue_content),
            purpose='assistants'
        )
        issue_descr = f'''{issue['number']}: {issue['title']}'''
        thread = self.openai.beta.threads.create(
            metadata={
                'issue': issue_descr,
                'run_id': os.getenv('GITHUB_RUN_ID', ''),
                'model': f'{self.assistant.model} t={self.assistant.temperature} top_p={self.assistant.top_p}',
            },
            messages=[{
                'role': 'user',
                'content': self.step_prompts[0].replace('<<Issue>>', issue_content.decode('utf-8')),
                'attachments': [{'file_id': issue_file.id, 'tools': [{'type': 'file_search'}]}]
            }]
        )
        print('Thread:', thread.id)
        thread_vstore_id = thread.tool_resources.file_search.vector_store_ids[0]
        self.openai.beta.vector_stores.update(
            thread_vstore_id,
            name=issue_descr,
            metadata={
                'thread_id': thread.id
            }
        )
        self.wait_vector_store(thread_vstore_id)

        self.run_assistant(thread, issue)
        for prompt in self.step_prompts[1:]:
            self.openai.beta.threads.messages.create(thread.id, role='user', content=prompt)
            self.run_assistant(thread, issue)

    def run_assistant(self, thread, issue):
        stream = self.openai.beta.threads.runs.create(
            assistant_id=self.assistant.id,
            thread_id=thread.id,
            instructions=self.system_prompt,
            stream=True
        )
        streams = [stream]
        while streams:
            stream = streams.pop(0)
            for event in stream:
                match event:
                    case ThreadMessageCompleted():
                        for c in event.data.content:
                            print('<<Message>>', c.text.value)
                    case ThreadRunRequiresAction():
                        streams.append(self.handle_tool_calls(issue['number'], thread, event))
                    case ThreadRunStepCompleted():
                        if isinstance(event.data.step_details, ToolCallsStepDetails):
                            for tool_call in event.data.step_details.tool_calls:
                                if isinstance(tool_call, FileSearchToolCall):
                                    print('<<File search>>')
                                    for result in tool_call.file_search.results:
                                        print(f'  {result.file_name}: {result.score}')

    def handle_tool_calls(self, issue_number: int, thread, event) -> object:
        def modify_repo(operation):
            if self.modify:
                response = operation()
                if not response.ok:
                    return f'''Failed: {response.json()['message']}'''
            return 'Ok'

        thread_vstore_id = thread.tool_resources.file_search.vector_store_ids[0]
        tool_outputs = []
        for tool in event.data.required_action.submit_tool_outputs.tool_calls:
            args = json.loads(tool.function.arguments)
            print(f'<<Tool call>>', tool.function.name, args)
            match tool.function.name:
                case 'search_github':
                    query = f'''repo:{self.search_repository} ({') OR ('.join(args['search_terms'])})'''
                    results = self.search_github(query, thread_vstore_id, curr_issue_number=issue_number)
                    if results:
                        result_lines = [f'Found {len(results)} results and attached as files to this thread:']
                        for issue_number, title, file in results:
                            print(f'  {issue_number}: {title}')
                            result_lines.append(f'{file.filename} => {issue_number}: {title}')
                        output = '\n'.join(result_lines)
                    else:
                        output = 'No results found.'
                    tool_outputs.append({'tool_call_id': tool.id, 'output': output})

                case 'get_external_content':
                    output = self.get_external_content(args['url'], args['description'], thread_vstore_id)
                    tool_outputs.append({'tool_call_id': tool.id, 'output': output})

                case 'add_issue_labels':
                    output = modify_repo(lambda: self.github_request(
                        'POST', f'/repos/{self.current_repository}/issues/{issue_number}/labels',
                        json={'labels': args['labels']}))
                    tool_outputs.append({'tool_call_id': tool.id, 'output': output})

                case 'set_issue_title':
                    output = modify_repo(lambda: self.github_request(
                        'PATCH', f'/repos/{self.current_repository}/issues/{issue_number}/labels',
                        json={'title': args['title']}))
                    tool_outputs.append({'tool_call_id': tool.id, 'output': output})

                case 'add_issue_comment':
                    output = modify_repo(lambda: self.github_request(
                        'POST', f'/repos/{self.current_repository}/issues/{issue_number}/comments',
                        json={'body': args['body']}))
                    tool_outputs.append({'tool_call_id': tool.id, 'output': output})

                case _:
                    raise ValueError('Unknown tool call')

        return self.openai.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=event.data.id,
            tool_outputs=tool_outputs,
            stream=True)

    def search_github(self, query: str, vstore_id: str, curr_issue_number=None, max_results=5) -> list:
        response = self.github_request('GET', '/search/issues', dict(q=query))
        if not response.ok:
            print(f'''Search failed: {response.json()['message']}''')
            return []

        issues = response.json()['items']
        results = []
        for issue in issues:
            issue_number = issue['number']
            if curr_issue_number is not None and issue_number >= curr_issue_number:
                continue

            if issue_number not in self.found_issues:  # Don't attach the same data twice
                issue_file = self.attach_file(vstore_id, f'ISSUE_{issue_number}.md',
                                              self.make_issue_content(issue, fetch_comments=True))
                self.found_issues[issue_number] = issue_file.id

            results.append((issue_number, issue['title'], issue_file))
            if len(results) >= max_results:
                break

        self.wait_vector_store(vstore_id)
        return results

    def get_external_content(self, url: str, description: str, vstore_id: str) -> str:
        response = requests.get(url)
        if response.ok:
            content_type = response.headers['content-type'].lower().split(';')[0]
            match content_type:
                case 'text/plain' | 'text/x-log': ext = 'txt'
                case 'text/html': ext = 'html'
                case 'text/markdown': ext = 'md'
                case 'image/png': ext = 'png'
                case 'image/jpeg': ext = 'jpg'
                case 'image/webp': ext = 'webp'
                case _:
                    return f'Failed: Unsupported content type ({content_type})'
            name = f'{description}.{ext}'
            self.attach_file(vstore_id, name, response.content)
            return f'Attached as "{name}"'
        else:
            return f'Failed: {response.text}'

    def attach_file(self, vstore_id: str, name: str, content: bytes) -> Any:
        file = self.openai.files.create(file=(name, content), purpose='assistants')
        self.openai.beta.vector_stores.files.create(vector_store_id=vstore_id, file_id=file.id)
        return file

    def make_issue_content(self, issue, fetch_comments=False, show_labels=True) -> bytes:
        f = io.StringIO()
        f.write(f'''### Title: {issue['title']}\n''')
        f.write(f'''### Author: {issue['user']['login']}\n''')
        f.write(f'''### State: {issue['state']}\n''')
        if show_labels:
            f.write(
                f'''### Labels: {','.join(label['name'] for label in issue['labels'])}\n''')
        f.write(f'''\n{issue['body']}\n''')

        if fetch_comments:
            response = self.github_request('GET', issue['comments_url'])
            if response.ok:
                for comment in response.json():
                    f.write(f'''### Comment by {comment['user']['login']}\n''')
                    f.write(f'''\n{comment['body']}\n''')

        return f.getvalue().encode('utf-8')

    def github_request(self, method, url, params=None, json=None):
        headers = {
            'X-GitHub-Api-Version': '2022-11-28',
            'Accept': 'application/vnd.github+json',
        }
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        if url[0] == '/':
            url = 'https://api.github.com' + url
        return requests.request(method, url, headers=headers, params=params, json=json)

    def wait_vector_store(self, vstore_id):
        vstore = self.openai.beta.vector_stores.retrieve(vstore_id)
        while vstore.status == 'in_progress':
            print('Waiting for vector store update.')
            time.sleep(1)
            vstore = self.openai.beta.vector_stores.retrieve(vstore_id)


if __name__ == '__main__':
    CodelldBot().handle_event()
