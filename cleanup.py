from argparse import ArgumentParser

from openai import NotFoundError, OpenAI

openai = OpenAI()

parser = ArgumentParser()
parser.add_argument('--all', action='store_true')
args = parser.parse_args()

for vstore in openai.beta.vector_stores.list(order='asc', limit=100):
    thread_id = vstore.metadata.get('thread_id')
    if thread_id and (vstore.status == 'expired' or args.all):
        print('Deleting', vstore.id)
        files = openai.beta.vector_stores.files.list(vstore.id, limit=100)
        while len(files.data) > 0:
            for file in files.data:
                print('  Deleting', file.id)
                try:
                    openai.files.delete(file.id)
                except NotFoundError:
                    pass
            files = openai.beta.vector_stores.files.list(vstore.id, after=files.data[-1].id, limit=100)

        openai.beta.vector_stores.delete(vstore.id)

        print('  Deleting', thread_id)
        openai.beta.threads.delete(thread_id)
