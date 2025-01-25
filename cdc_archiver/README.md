## CDC Archive Archiver

Archives a CDC archive search query. Websites are archived via Webrecorder and files (pdf, xlsx) are downloaded directly.

- `conda create --name cdc python=3.11`
- `conda activate cdc`
- `pip install -r requirements.txt`
- `./cdc_archiver.sh {KEYWORD}` e.g. `./cdc_archiver.sh nonbinary`

When the scripts are done and everything is archived:

- `./kill_wayback_server.sh`

WARC archive is under `./{KEYWORD}/collections/cdc-archive-{KEYWORD}/archive/` and files are under `./{KEYWORD}/files`.

## Todos

- [ ] there are probably more file endings to be catched
- [ ] needs testing with longer search results
