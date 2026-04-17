import importlib
import importlib.util
import sys

modules = ['praw','pytrends','requests','newsapi','vaderSentiment','pandas','dotenv']
missing = []
for m in modules:
    try:
        spec = importlib.util.find_spec(m)
        if spec is None:
            missing.append(m)
    except Exception:
        missing.append(m)

print('MISSING_PACKAGES:' + (','.join(missing)))

proj_mods = [
    'data_collectors.reddit_search',
    'data_collectors.trends',
    'data_collectors.news',
    'analysis.sentiment',
    'analysis.aggregator',
    'utils.config',
    'utils.storage'
]
proj_errors = []
for m in proj_mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        proj_errors.append(f"{m}::{e.__class__.__name__}::{e}")

if proj_errors:
    print('PROJECT_IMPORT_ERRORS:')
    for e in proj_errors:
        print(e)
else:
    print('PROJECT_IMPORTS_OK')

print('PYTHON:' + sys.version.replace('\n',' '))
