import io
import json
import os
import time

import requests
from openai import OpenAI
from openai.types.beta.assistant_stream_event import ThreadMessageCompleted, ThreadRunRequiresAction


class CodelldBot:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.openai = OpenAI()
        self.current_repository = os.getenv('GITHUB_REPOSITORY')
        self.search_repository = os.getenv('SEARCH_REPOSITORY') or self.current_repository
        self.modify = bool(self.token and os.getenv('MODIFY'))

    def handle_event(self):

        with open(os.getenv('GITHUB_EVENT_PATH'), 'rb') as f:
            event = json.load(f)

        match os.getenv('GITHUB_EVENT_NAME'):
            case 'issues':
                issue = event['issue']['number']
            case 'workflow_dispatch':
                issue_number = int(event['inputs']['issue_number'])
                response = self.github_request(
                    'GET', f'/repos/{self.current_repository}/issues/{issue_number}')
                if not response.ok:
                    raise Exception(f'''Could not fetch issue: {response['message']}''')
                issue = response.json()

        assistant = self.openai.beta.assistants.retrieve(
            os.getenv('ASSISTANT_ID'))

        issue_file = self.openai.files.create(
            file=('BUG_REPORT.md', self.make_issue_content(
                issue, show_labels=False)),
            purpose='assistants'
        )

        thread = self.openai.beta.threads.create(
            metadata={
                'issue': f'''{issue['number']}: {issue['title']}''',
                'run_id': os.getenv('GITHUB_RUN_ID', '#'),
                'model': f'{assistant.model} t={assistant.temperature} top_p={assistant.top_p}',
            },
            messages=[{
                'role': 'user',
                'content': 'We have a new issue report (attached as BUG_REPORT.md)',
                'attachments': [{'file_id': issue_file.id, 'tools': [{'type': 'file_search'}]}]
            }]
        )
        print('Thread:', thread.id)

        self.wait_vector_store(
            thread.tool_resources.file_search.vector_store_ids[0])

        stream = self.openai.beta.threads.runs.create(
            assistant_id=assistant.id,
            thread_id=thread.id,
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

    def handle_tool_calls(self, issue_number: int, thread, event) -> object:
        def modify_repo(operation):
            if self.modify:
                response = operation()
                if not response.ok:
                    return f'''Failed: {response.json()['message']}'''
            return 'Ok'

        tool_outputs = []
        for tool in event.data.required_action.submit_tool_outputs.tool_calls:
            args = json.loads(tool.function.arguments)
            print(f'<<Tool call>>', tool.function.name, args)
            match tool.function.name:
                case 'search_github':
                    query = f'''repo:{self.search_repository} {args['query']}'''
                    thread_vstore_id = thread.tool_resources.file_search.vector_store_ids[0]
                    output = self.search_github(
                        query, thread_vstore_id, exclude=[issue_number])
                    tool_outputs.append(
                        {'tool_call_id': tool.id, 'output': output})
                case 'add_issue_labels':
                    output = modify_repo(lambda: self.github_request(
                        'POST', f'/repos/{self.current_repository}/issues/{issue_number}/labels',
                        json={'labels': args['labels']}))
                    tool_outputs.append(
                        {'tool_call_id': tool.id, 'output': output})
                case 'set_issue_title':
                    output = modify_repo(lambda: self.github_request(
                        'PATCH', f'/repos/{self.current_repository}/issues/{issue_number}/labels',
                        json={'title': args['title']}))
                    tool_outputs.append(
                        {'tool_call_id': tool.id, 'output': output})
                case 'add_issue_comment':
                    output = modify_repo(lambda: self.github_request(
                        'POST', f'/repos/{self.current_repository}/issues/{issue_number}/comments',
                        json={'body': args['body']}))
                    tool_outputs.append(
                        {'tool_call_id': tool.id, 'output': output})

        return self.openai.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=event.data.id,
            tool_outputs=tool_outputs,
            stream=True)

    def search_github(self, query: str, vstore_id: str, exclude: list = [], max_results=5) -> str:
        response = self.github_request('GET', '/search/issues', dict(q=query))
        if not response.ok:
            return f'''Search failed: {response.json()['message']}'''

        issues = response.json()['items']
        total_results = len(issues)
        result_lines = []
        for issue in issues:
            issue_number = issue['number']
            if issue_number in exclude:
                total_results -= 1
                continue
            issue_file = self.openai.files.create(
                file=(f'ISSUE_{issue_number}.md', self.make_issue_content(issue, fetch_comments=True)),
                purpose='assistants'
            )
            self.openai.beta.vector_stores.files.create(
                vector_store_id=vstore_id,
                file_id=issue_file.id,
            )
            result_lines.append(
                f'Issue number: {issue_number}, file name: {issue_file.filename}')
            if len(result_lines) >= max_results:
                break

        self.wait_vector_store(vstore_id)

        if len(result_lines) > 0:
            summary = f'Found {total_results} issues, of which top {len(result_lines)} were attached as files to this thread:'
            return '\n'.join([summary] + result_lines)
        else:
            return 'Search produced no results.'

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
